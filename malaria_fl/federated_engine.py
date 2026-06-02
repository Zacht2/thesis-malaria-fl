"""
federated_engine.py  (v2 — robust, real-data)
==============================================
Core FL simulation engine wired to real_data_pipeline.py and
data_degradation.py.

NEW IN V2:
  1. Real data loading via real_data_pipeline.build_all_clinic_data()
  2. Clinic-specific data degradation (heterogeneous non-IID conditions)
  3. Confidence-score-weighted FedAvg aggregation
  4. Noise injection during local training
  5. Missing-data masking (3 indicator features appended)
  6. Quality-aware Shapley valuation

FL PROTOCOL (one round):
  1. Server broadcasts global model weights
  2. Each clinic:
       a. Applies degradation to its data (noise + missing indicators)
       b. Trains local model (GBT fine-tuned from global)
       c. Computes its own confidence score
       d. Sends model + confidence + metrics to server (no raw data)
  3. Server performs Quality-Weighted FedAvg:
       weight_i = n_train_i * confidence_i   (Chai et al. 2020)
  4. LOO Shapley computed over degraded client data
  5. Governance issues IP tokens (Shapley + volume + quality + rarity)

REFERENCES:
  FedAvg: McMahan et al. (2017) Communication-Efficient Learning
  Weighted FedAvg: Chai et al. (2020) Towards Tailed Query Detection
  Missing mask: Mattei & Frellsen (2019) MIWAE, ICML
  Noise injection: Deng et al. (2021) Astraea Self-Balancing FL
"""

import numpy as np
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split

from models import (
    make_model, train_local_model, evaluate_model,
    aggregate_models_fedavg, get_feature_importances, ablation_study
)
from governance import (
    IPTokenLedger, ShapleyApproximator, ParticipationIncentiveManager
)
from real_data_pipeline import (
    build_all_clinic_data, build_feature_matrix,
    get_feature_names, CLINICS, N_WEEKS
)
from data_degradation import (
    degrade_clinic_data, get_feature_names_with_masks,
    DEGRADATION_PROFILES, N_TOTAL_WITH_MASK
)

warnings.filterwarnings("ignore")

N_FEATURES = N_TOTAL_WITH_MASK   # 55 after masking (13 env + 6 drug + 33 img + 3 masks)


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED CLIENT  (v2)
# ─────────────────────────────────────────────────────────────────────────────

class FederatedClient:
    """
    Represents one clinic node in the federated network.

    Each round the client:
      1. Receives global model from server
      2. Applies degradation to its data (fresh each round — noise varies)
      3. Trains locally
      4. Reports model + confidence score + metrics (no raw data)
    """

    def __init__(self, clinic_id: int, clinic_data: dict):
        self.clinic_id   = clinic_id
        self.clinic_info = clinic_data["clinic_info"]
        self.name        = self.clinic_info["name"]
        self.profile     = DEGRADATION_PROFILES[clinic_id]

        # Build CLEAN feature matrix (47 features)
        X_clean, y_clean = build_feature_matrix(clinic_data)
        self.X_clean = X_clean
        self.y_clean = y_clean

        # Drug resistance data for governance rarity scoring
        self.drug_X = clinic_data["drug_X"]

        # Time-aware 80/20 split on clean data
        self.split_idx = int(N_WEEKS * 0.80)   # 124 train, 32 test
        self.n_train   = self.split_idx
        self.n_test    = N_WEEKS - self.split_idx

        self.global_model     = None
        self.local_model      = None
        self.last_confidence  = {}
        self.last_degraded    = {}

    def receive_global_model(self, global_model):
        self.global_model = global_model

    def local_train(self,
                    all_n_train: dict,
                    round_num:   int = 1,
                    inject_noise: bool = True) -> dict:
        """
        Applies degradation, trains locally, returns update + metadata.
        No raw feature vectors leave this function — only model + scalars.
        """
        t0 = time.time()

        # Apply degradation (fresh noise each round)
        degraded = degrade_clinic_data(
            X           = self.X_clean,
            y           = self.y_clean,
            clinic_id   = self.clinic_id,
            split_idx   = self.split_idx,
            all_n_train = all_n_train,
            round_num   = round_num,
            inject_noise= inject_noise,
        )
        self.last_degraded   = degraded
        self.last_confidence = degraded["confidence"]

        X_train = degraded["X_train"]   # (124, 50)
        y_train = degraded["y_train"]
        X_test  = degraded["X_test"]    # (32, 50)
        y_test  = degraded["y_test"]

        # Local training (starts from global model weights)
        self.local_model = train_local_model(
            X_train, y_train,
            self.clinic_id,
            base_model=self.global_model,
        )

        train_time    = time.time() - t0
        train_metrics = evaluate_model(self.local_model, X_train, y_train)
        test_metrics  = evaluate_model(self.local_model, X_test,  y_test)

        return {
            "clinic_id":        self.clinic_id,
            "clinic_name":      self.name,
            "model":            self.local_model,
            "n_train":          self.n_train,
            "confidence":       self.last_confidence,
            "train_metrics":    train_metrics,
            "test_metrics":     test_metrics,
            "train_time_s":     round(train_time, 2),
            # Test data passed for server-side evaluation
            # (this is the only data crossing the boundary — not training data)
            "_X_test":          X_test,
            "_y_test":          y_test,
        }

    def get_info(self) -> dict:
        return {
            "clinic_id":     self.clinic_id,
            "name":          self.name,
            "n_train":       self.n_train,
            "n_test":        self.n_test,
            "pos_rate":      float(self.y_clean[:self.split_idx].mean()),
            "missing_rate":  self.profile["missing_rate"],
            "img_quality":   self.profile["img_quality"],
            "facility_type": self.clinic_info["facility_type"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED SERVER  (v2)
# ─────────────────────────────────────────────────────────────────────────────

class FederatedServer:
    """
    Central aggregation server.
    Holds no raw patient data. Only receives model parameters + scalars.

    Implements Quality-Weighted FedAvg:
        weight_i = n_train_i * confidence_i

    Reference: Chai et al. (2020) "Towards Tailed Query Detection in
    Federated Learning for Drug Discovery"
    """

    def __init__(self, n_clients: int = 10):
        self.n_clients      = n_clients
        self.global_model   = None
        self.round_history: List[dict] = []
        self.shapley_approx = ShapleyApproximator()

        # Server-side reference dataset for knowledge distillation
        # Synthetic: 200 samples, no real patient data
        rng = np.random.default_rng(999)
        self._ref_X = rng.random((200, N_FEATURES)).astype(np.float32)
        self._ref_y = (rng.random(200) > 0.78).astype(int)

    def initialise_global_model(self):
        """Cold-start on synthetic reference data."""
        print("  [SERVER] Initialising global model (cold start)...")
        self.global_model = make_model(random_state=42)
        self.global_model.fit(self._ref_X, self._ref_y)
        print(f"  [SERVER] Global model ready. "
              f"Input dim: {N_FEATURES} features "
              f"(53 real + 3 missing indicators).")

    def aggregate(self, client_results: List[dict]) -> dict:
        """
        Quality-Weighted FedAvg.

        Weight = n_train * confidence_score
        This down-weights clinics with high missingness or poor image quality,
        giving their model updates proportionally less influence on the global
        model. Clean, complete data = more influence.

        References:
          McMahan et al. (2017) — standard FedAvg (n_train weighting)
          Chai et al. (2020)    — quality-aware weighting extension
        """
        models      = [r["model"]   for r in client_results]
        n_trains    = [r["n_train"] for r in client_results]
        confidences = [r["confidence"]["confidence_score"] for r in client_results]

        # Quality-weighted: n_train * confidence
        weights = [n * c for n, c in zip(n_trains, confidences)]

        self.global_model = aggregate_models_fedavg(
            models, weights, self._ref_X, self._ref_y
        )

        # Evaluate global on all clients' test sets
        all_X = np.vstack([r["_X_test"] for r in client_results])
        all_y = np.concatenate([r["_y_test"] for r in client_results])
        return evaluate_model(self.global_model, all_X, all_y)

    def compute_loo_shapley(self,
                              client_results: List[dict],
                              all_clients: Dict[int, FederatedClient],
                              global_auc: float) -> Dict[int, float]:
        """
        LOO Shapley using quality-weighted coalitions.

        For each clinic i:
          1. Form coalition C = all clients except i
          2. Aggregate C with quality-weighted FedAvg
          3. Evaluate on clinic i's TEST data
          4. Shapley_i = global_auc - auc(coalition without i)

        A positive Shapley score means "removing this clinic hurts performance" —
        the clinic is genuinely contributing unique signal.
        """
        print("  [SERVER] Computing quality-weighted Shapley values (LOO)...")
        loo_metrics: Dict[int, dict] = {}

        for leave_out_id in sorted(all_clients.keys()):
            coalition = [r for r in client_results
                         if r["clinic_id"] != leave_out_id]
            if not coalition:
                loo_metrics[leave_out_id] = {"auc_roc": 0.5}
                continue

            c_models  = [r["model"]   for r in coalition]
            c_trains  = [r["n_train"] for r in coalition]
            c_conf    = [r["confidence"]["confidence_score"] for r in coalition]
            c_weights = [n * c for n, c in zip(c_trains, c_conf)]

            loo_model = aggregate_models_fedavg(
                c_models, c_weights, self._ref_X, self._ref_y
            )
            client     = all_clients[leave_out_id]
            X_test_lo  = client.last_degraded["X_test"]
            y_test_lo  = client.last_degraded["y_test"]
            loo_metrics[leave_out_id] = evaluate_model(loo_model, X_test_lo, y_test_lo)

        return self.shapley_approx.compute_loo_shapley(
            client_metrics = loo_metrics,
            global_metric  = global_auc,
            metric_key     = "auc_roc",
        )


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED SIMULATION ORCHESTRATOR  (v2)
# ─────────────────────────────────────────────────────────────────────────────

class FederatedSimulation:
    """
    Orchestrates the full FL simulation.
    """

    def __init__(self,
                 pf8_path:    str = "",
                 map_path:    str = "",
                 image_dir:   str = "/tmp/malaria_images",
                 era5_dir:    str = None,
                 data_dir:    str = "/tmp/clinic_data_real_v4",
                 n_rounds:    int = 5,
                 inject_noise: bool = True,
                 verbose:     bool = True):

        self.pf8_path     = pf8_path
        self.map_path     = map_path
        self.image_dir    = image_dir
        self.era5_dir     = era5_dir
        self.data_dir     = data_dir
        self.n_rounds     = n_rounds
        self.inject_noise = inject_noise
        self.verbose      = verbose

        self.ledger      = IPTokenLedger()
        self.incentives  = ParticipationIncentiveManager(n_rounds)
        self.results_log: List[dict] = []

    def setup(self):
        print("\n" + "="*65)
        print("  MALARIA FL FRAMEWORK  (v2 — robust + real data)")
        print("="*65)
        print(f"  Clinics      : {len(CLINICS)}")
        print(f"  Rounds       : {self.n_rounds}")
        print(f"  Features     : {N_FEATURES} "
              f"(env=13, drug=6, img=34, missing_masks=3)")
        print(f"  Weeks/clinic : {N_WEEKS} (2019-2021)")
        print(f"  Noise inject : {self.inject_noise}")
        print("="*65)

        # Load real data
        print("\n  Loading real data sources...")
        all_data = build_all_clinic_data(
            pf8_path      = self.pf8_path,
            map_path      = self.map_path,
            image_dir     = self.image_dir,
            era5_dir      = self.era5_dir,
            output_dir    = self.data_dir,
            force_rebuild = False,
        )

        # Instantiate clients
        self.clients: Dict[int, FederatedClient] = {}
        for cid, data in all_data.items():
            self.clients[cid] = FederatedClient(cid, data)

        # Server
        self.server = FederatedServer(n_clients=len(self.clients))
        self.server.initialise_global_model()

        # Print clinic overview
        print(f"\n  {'Clinic':<42} {'Pos%':>5} {'Miss%':>6} {'ImgQ':>6} {'Type'}")
        print(f"  {'-'*72}")
        for cid, client in sorted(self.clients.items()):
            info = client.get_info()
            print(f"  [{cid}] {info['name']:<40} "
                  f"{info['pos_rate']:>4.0%} "
                  f"{info['missing_rate']:>5.0%} "
                  f"{info['img_quality']:>6.2f} "
                  f"  {info['facility_type']}")

        return all_data

    def run(self):
        self.setup()

        print(f"\n{'='*65}")
        print("  STARTING FEDERATED TRAINING")
        print(f"{'='*65}")

        all_n_train = {cid: c.n_train for cid, c in self.clients.items()}

        for rnd in range(1, self.n_rounds + 1):
            self._run_round(rnd, all_n_train)

        self._final_report()
        return self.results_log

    def _run_round(self, round_num: int, all_n_train: dict):
        print(f"\n{'─'*65}")
        print(f"  ROUND {round_num}/{self.n_rounds}")
        print(f"{'─'*65}")

        # Broadcast global model
        for client in self.clients.values():
            client.receive_global_model(self.server.global_model)
            self.incentives.record_participation(client.clinic_id, round_num)

        # Local training
        print(f"\n  Local training ({len(self.clients)} clinics)...")
        client_results = []
        for cid in sorted(self.clients.keys()):
            client = self.clients[cid]
            result = client.local_train(
                all_n_train  = all_n_train,
                round_num    = round_num,
                inject_noise = self.inject_noise,
            )
            client_results.append(result)

            conf = result["confidence"]["confidence_score"]
            acc  = result["test_metrics"]["accuracy"]
            auc  = result["test_metrics"]["auc_roc"]
            f1   = result["test_metrics"]["f1"]
            print(f"    [{cid}] {client.name:<42} "
                  f"conf={conf:.2f}  acc={acc:.3f}  auc={auc:.3f}  f1={f1:.3f}")

        # Quality-weighted aggregation
        print(f"\n  [SERVER] Quality-Weighted FedAvg...")
        # Print weights for transparency
        total_w = sum(r["n_train"] * r["confidence"]["confidence_score"]
                      for r in client_results)
        for r in sorted(client_results, key=lambda x: x["clinic_id"]):
            w = r["n_train"] * r["confidence"]["confidence_score"]
            print(f"    [{r['clinic_id']}] weight = "
                  f"{r['n_train']} × {r['confidence']['confidence_score']:.3f} "
                  f"= {w:.1f}  ({w/total_w:.1%} of total)")

        global_metrics = self.server.aggregate(client_results)
        print(f"\n  [SERVER] Global model: "
              f"acc={global_metrics['accuracy']:.4f}  "
              f"auc={global_metrics['auc_roc']:.4f}  "
              f"f1={global_metrics['f1']:.4f}  "
              f"sens={global_metrics.get('sensitivity',0):.4f}")

        # Shapley
        shapley_scores = self.server.compute_loo_shapley(
            client_results, self.clients, global_metrics["auc_roc"]
        )

        # Governance
        clinic_data_info   = {cid: c.get_info() for cid, c in self.clients.items()}
        local_metrics_map  = {r["clinic_id"]: r["test_metrics"]
                               for r in client_results}
        drug_data_map      = {cid: c.drug_X for cid, c in self.clients.items()}
        confidence_map     = {r["clinic_id"]: r["confidence"]
                               for r in client_results}

        self.ledger.log_round_contributions(
            round_num                = round_num,
            global_metrics           = global_metrics,
            shapley_scores           = shapley_scores,
            clinic_data_info         = clinic_data_info,
            local_metrics_per_clinic = local_metrics_map,
            drug_data_per_clinic     = drug_data_map,
            verbose                  = True,
        )

        self.results_log.append({
            "round":          round_num,
            "global_metrics": global_metrics,
            "shapley_scores": shapley_scores,
            "local_metrics":  local_metrics_map,
            "confidence":     confidence_map,
        })

    def _final_report(self):
        print(f"\n{'='*65}")
        print("  FEDERATED TRAINING COMPLETE — FINAL REPORT")
        print(f"{'='*65}")

        # Global model performance over rounds
        print(f"\n  GLOBAL MODEL PERFORMANCE OVER {self.n_rounds} ROUNDS:")
        print(f"  {'Round':<8} {'Accuracy':>10} {'AUC-ROC':>10} "
              f"{'F1':>8} {'Sensitivity':>13}")
        print(f"  {'-'*53}")
        for entry in self.results_log:
            r  = entry["round"]
            gm = entry["global_metrics"]
            print(f"  {r:<8} {gm['accuracy']:>10.4f} {gm['auc_roc']:>10.4f} "
                  f"{gm['f1']:>8.4f} {gm.get('sensitivity',0):>13.4f}")

        # Data quality vs. Shapley contribution
        last = self.results_log[-1]
        print(f"\n  DATA QUALITY vs SHAPLEY CONTRIBUTION:")
        print(f"  {'Clinic':<42} {'Conf':>6} {'Shapley':>9} {'Interpretation'}")
        print(f"  {'-'*72}")
        for cid in sorted(last["shapley_scores"].keys()):
            sv   = last["shapley_scores"][cid]
            conf = last["confidence"][cid]["confidence_score"]
            name = self.clients[cid].name[:40]
            miss = DEGRADATION_PROFILES[cid]["missing_rate"]
            tag  = ("  HIGH MISS" if miss >= 0.20 else
                    "  LOW MISS"  if miss <= 0.05 else "")
            interp = ("Beneficial" if sv > 0.02 else
                      "Marginal"   if sv > -0.01 else "Detrimental")
            print(f"  {name:<42} {conf:>6.3f} {sv:>+9.4f}  {interp}{tag}")

        # IP token balances
        summary = self.ledger.get_ledger_summary()
        print(f"\n  FINAL IP TOKEN BALANCES:")
        print(f"  {'Clinic':<42} {'Tokens':>8} {'Share%':>8} {'Conf':>6}")
        print(f"  {'-'*68}")
        total_tokens = summary["total_tokens_issued"]
        for cid in sorted(summary["token_balances"].keys()):
            tokens = summary["token_balances"][cid]
            share  = (tokens / total_tokens * 100) if total_tokens > 0 else 0
            conf   = last["confidence"][cid]["confidence_score"]
            lm     = self.incentives.get_loyalty_multiplier(cid, self.n_rounds)
            em     = self.incentives.get_early_adopter_bonus(cid)
            adj    = round(tokens * lm * em, 2)
            name   = self.clients[cid].name[:40]
            print(f"  {name:<42} {adj:>8.2f} {share:>7.1f}% {conf:>6.3f}")

        print(f"\n  Ledger Integrity: "
              f"{'VALID' if summary['chain_intact'] else 'TAMPERED'}")
        print(f"  Total tokens issued: {total_tokens:.2f}")

        # Ablation study
        print(f"\n  MODALITY ABLATION STUDY:")
        self._run_ablation()

        # Feature importance
        print(f"\n  TOP-10 PREDICTIVE FEATURES (Final Global Model):")
        self._print_feature_importance()

        # Robustness summary
        print(f"\n  ROBUSTNESS ANALYSIS:")
        print(f"  High-quality clinics (conf > 0.85):")
        hq = [(cid, last["confidence"][cid]["confidence_score"],
               last["shapley_scores"][cid])
              for cid in range(10)
              if last["confidence"][cid]["confidence_score"] > 0.85]
        lq = [(cid, last["confidence"][cid]["confidence_score"],
               last["shapley_scores"][cid])
              for cid in range(10)
              if last["confidence"][cid]["confidence_score"] <= 0.75]
        for cid, conf, sv in sorted(hq, key=lambda x: -x[1]):
            print(f"    {self.clients[cid].name:<42} conf={conf:.3f}  sv={sv:+.4f}")
        print(f"  Low-quality clinics (conf <= 0.75):")
        for cid, conf, sv in sorted(lq, key=lambda x: x[1]):
            print(f"    {self.clients[cid].name:<42} conf={conf:.3f}  sv={sv:+.4f}")

        hq_sv = np.mean([x[2] for x in hq]) if hq else 0
        lq_sv = np.mean([x[2] for x in lq]) if lq else 0
        print(f"\n  Mean Shapley — high quality: {hq_sv:+.4f}")
        print(f"  Mean Shapley — low quality:  {lq_sv:+.4f}")
        print(f"  (Quality-weighted FedAvg: low-quality clinics receive "
              f"proportionally less weight, limiting their impact on global model)")
        print(f"\n{'='*65}\n")

    def _run_ablation(self):
        """Ablation on pooled degraded data from all clinics."""
        all_Xtr, all_ytr = [], []
        all_Xte, all_yte = [], []
        for client in self.clients.values():
            d = client.last_degraded
            all_Xtr.append(d["X_train"]); all_ytr.append(d["y_train"])
            all_Xte.append(d["X_test"]);  all_yte.append(d["y_test"])
        X_tr = np.vstack(all_Xtr); y_tr = np.concatenate(all_ytr)
        X_te = np.vstack(all_Xte); y_te = np.concatenate(all_yte)

        results = ablation_study(X_tr, y_tr, X_te, y_te)
        print(f"  {'Config':<25} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>8}")
        print(f"  {'-'*55}")
        for name, m in results.items():
            if "error" in m:
                print(f"  {name:<25} ERROR: {m['error']}")
            else:
                print(f"  {name:<25} {m['accuracy']:>10.4f} "
                      f"{m['auc_roc']:>10.4f} {m['f1']:>8.4f}")

    def _print_feature_importance(self):
        feat_names = get_feature_names_with_masks()
        imps = get_feature_importances(self.server.global_model, feat_names)
        if not imps:
            print("  (Not available for this model type)")
            return
        for rank, (fname, imp) in enumerate(list(imps.items())[:10], 1):
            bar = "█" * int(imp * 80)
            print(f"  {rank:>2}. {fname:<28} {imp:.4f}  {bar}")

        # Highlight if missing indicators appear in top features
        mask_feats = [(f, v) for f, v in imps.items() if "mask_" in f]
        if mask_feats:
            print(f"\n  Missing-indicator features in importance ranking:")
            for f, v in mask_feats:
                print(f"    {f}: {v:.4f}  "
                      f"(model is using absence-of-data as a signal)")

    def save_results(self, path: str) -> str:
        """Pickle the full results log to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "rounds":        self.results_log,
                "clinic_info":   {cid: c.get_info() for cid, c in self.clients.items()},
                "global_model":  self.server.global_model,
            }, f)
        print(f"  💾 Results saved → {path}")
        return path

    def export_governance_csv(self, path: str):
        """Export the IP token ledger to CSV."""
        import pandas as pd
        if not self.ledger.chain:
            print("  ⚠️  No ledger entries to export.")
            return pd.DataFrame()
        rows = []
        for block in self.ledger.chain:
            for tx in block.get("transactions", []):
                rows.append(tx)
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"  📋 Ledger exported → {path}")
        return df


if __name__ == "__main__":
    sim = FederatedSimulation(
        pf8_path     = "/Users/zacharythurston/Desktop/malaria_fl/Pf8-samples.csv",
        map_path     = "/Users/zacharythurston/Desktop/malaria_fl/Subnational_Unit-data.csv",
        image_dir    = "/tmp/malaria_images",
        era5_dir     = "/Users/zacharythurston/Desktop/malaria_fl",
        data_dir     = "/tmp/clinic_data_real_v4",
        n_rounds     = 5,
        inject_noise = True,
        verbose      = False,
    )
    sim.run()
