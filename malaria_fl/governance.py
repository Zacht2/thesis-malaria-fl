"""
governance.py
=============
Decentralised Governance and IP Attribution Module.

Implements the "Rule Book" described in thesis Section 5.3:

  1. Shapley Value Approximation
     — Measures the MARGINAL contribution of each clinic to the global model.
     — Prevents the "substitution effect": a clinic with rare genomic data
       gets more credit than one whose data is redundant.

  2. IP Token Ledger
     — Immutable append-only log of every training round's contributions.
     — Block-style structure (each round = one block) with SHA-256 chaining.
     — Tokens accumulate across rounds; represent fractional IP ownership.

  3. Valuation Formula (from thesis):
     Value = α × Shapley_Score + β × Data_Volume_Score + γ × Data_Quality_Score
     Default: α=0.60, β=0.20, γ=0.20

  4. Participation Incentive Multiplier
     — Clinics that join early rounds get a small bonus (encourages cold-start)
     — Clinics that contribute rare drug resistance data get a rarity bonus

Reference: Han et al. (2025) "Data valuation for vertical federated learning:
           A model-free and privacy-preserving method" (MIS Quarterly)
"""

import hashlib
import json
import time
import numpy as np
from typing import Dict, List, Optional
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# SHAPLEY VALUE APPROXIMATOR
# ---------------------------------------------------------------------------

class ShapleyApproximator:
    """
    Computes approximate Shapley values for FL client contributions.

    Full Shapley requires 2^N coalition evaluations — infeasible for N=10.
    We use the Leave-One-Out (LOO) approximation, which is standard in FL:

        φ_i ≈ v(S_all) - v(S_all \ {i})

    where v(S) = AUC-ROC of the global model trained without client i.

    For a more rigorous approximation when N is small (≤ 10), we also
    implement the permutation sampling method (Ghorbani & Zou, 2019).
    """

    def __init__(self, n_permutations: int = 50):
        self.n_permutations = n_permutations

    def compute_loo_shapley(self,
                             client_metrics: Dict[int, dict],
                             global_metric: float,
                             metric_key: str = "auc_roc") -> Dict[int, float]:
        """
        Leave-One-Out Shapley approximation.

        Args:
            client_metrics: {clinic_id: {"auc_roc": ..., "accuracy": ..., ...}}
                            Each entry = performance of global model trained
                            WITHOUT that client.
            global_metric:  Performance of full global model (all clients).
            metric_key:     Which metric to use for valuation.

        Returns:
            {clinic_id: shapley_value}  (positive = beneficial contribution)
        """
        shapley = {}
        for cid, metrics in client_metrics.items():
            if "error" in metrics:
                shapley[cid] = 0.0
                continue
            loo_score = metrics.get(metric_key, global_metric)
            # Shapley ≈ marginal gain from including clinic i
            shapley[cid] = float(global_metric - loo_score)

        # Normalise so values sum to the total gain over random baseline
        total = sum(abs(v) for v in shapley.values())
        if total > 0:
            shapley = {k: v / total for k, v in shapley.items()}

        return shapley

    def compute_permutation_shapley(self,
                                     eval_fn,
                                     client_ids: List[int],
                                     n_permutations: int = None) -> Dict[int, float]:
        """
        Permutation sampling Shapley (more accurate than LOO for small N).

        eval_fn(subset: list) -> float
            Function that evaluates the coalition's model performance.
            IMPORTANT: This function is called on a synthetic validation set,
            NOT on any clinic's private data -- privacy is preserved.

        This is the method described in the thesis for the governance module.
        """
        if n_permutations is None:
            n_permutations = self.n_permutations

        n = len(client_ids)
        marginals = {cid: [] for cid in client_ids}

        rng = np.random.default_rng(42)

        for _ in range(n_permutations):
            perm   = rng.permutation(client_ids).tolist()
            subset = []
            prev_val = eval_fn([])   # Empty coalition baseline

            for cid in perm:
                subset.append(cid)
                curr_val = eval_fn(subset)
                marginals[cid].append(curr_val - prev_val)
                prev_val = curr_val

        shapley = {cid: float(np.mean(vals)) for cid, vals in marginals.items()}

        # Normalise
        total = sum(abs(v) for v in shapley.values())
        if total > 0:
            shapley = {k: v / total for k, v in shapley.items()}

        return shapley


# ---------------------------------------------------------------------------
# VALUATION ENGINE
# ---------------------------------------------------------------------------

class ValuationEngine:
    """
    Computes the final IP Token award for each clinic per round.

    Formula (thesis Section 5.3):
        Value_i = α × Shapley_i + β × Volume_i + γ × Quality_i + δ × Rarity_i

    Where:
        Shapley_i = normalised marginal contribution (Shapley approximation)
        Volume_i  = normalised data volume (n_training_samples / max_samples)
        Quality_i = normalised local model AUC-ROC
        Rarity_i  = rarity bonus for unique/rare drug resistance data
    """

    ALPHA = 0.60   # Shapley weight (thesis: marginal value emphasis)
    BETA  = 0.20   # Volume weight
    GAMMA = 0.15   # Quality weight
    DELTA = 0.05   # Rarity bonus weight

    @staticmethod
    def compute_volume_score(n_samples: int, max_samples: int) -> float:
        """Log-normalised volume score to prevent large clinics dominating."""
        if max_samples == 0:
            return 0.0
        return float(np.log1p(n_samples) / np.log1p(max_samples))

    @staticmethod
    def compute_quality_score(metrics: dict) -> float:
        """Combined quality score from accuracy + AUC-ROC."""
        acc = metrics.get("accuracy", 0.5)
        auc = metrics.get("auc_roc",  0.5)
        f1  = metrics.get("f1",       0.5)
        return float(0.4 * acc + 0.4 * auc + 0.2 * f1)

    @staticmethod
    def compute_rarity_score(drug_X: np.ndarray) -> float:
        """
        Rarity bonus: clinics with high pfk13 allele frequencies (drug resistant
        strains) are rarer and more valuable for global surveillance.
        pfk13 = first drug resistance feature.
        """
        pfk13_freq = float(drug_X[:, 0].mean())
        # Rarity is highest for extreme values (very high = emerging resistance)
        rarity = 2 * abs(pfk13_freq - 0.10)   # Baseline population average ~10%
        return float(np.clip(rarity, 0, 1))

    def compute_token_award(self,
                             clinic_id: int,
                             shapley_score: float,
                             n_samples: int,
                             max_samples: int,
                             local_metrics: dict,
                             drug_X: np.ndarray) -> dict:
        """
        Returns full valuation breakdown for one clinic in one round.
        """
        vol_score     = self.compute_volume_score(n_samples, max_samples)
        quality_score = self.compute_quality_score(local_metrics)
        rarity_score  = self.compute_rarity_score(drug_X)

        # Clip shapley to [0, 1] for token calculation
        # (negative Shapley = clinic hurt the model; still gets small base award)
        shapley_clamped = max(0.0, shapley_score)

        raw_value = (
            self.ALPHA * shapley_clamped +
            self.BETA  * vol_score +
            self.GAMMA * quality_score +
            self.DELTA * rarity_score
        )

        # Scale to 0–100 IP Tokens per round
        tokens = round(raw_value * 100, 4)

        return {
            "clinic_id":       clinic_id,
            "tokens_awarded":  tokens,
            "shapley_score":   round(shapley_score, 6),
            "volume_score":    round(vol_score, 4),
            "quality_score":   round(quality_score, 4),
            "rarity_score":    round(rarity_score, 4),
            "n_samples":       n_samples,
            "local_accuracy":  round(local_metrics.get("accuracy", 0), 4),
            "local_auc":       round(local_metrics.get("auc_roc", 0), 4),
        }


# ---------------------------------------------------------------------------
# IP TOKEN LEDGER (Blockchain-style)
# ---------------------------------------------------------------------------

class IPTokenLedger:
    """
    Append-only ledger recording all IP token transactions.

    Structure:
        Genesis Block → Round 1 Block → Round 2 Block → ...

    Each block contains:
        - index, timestamp, previous_hash, block_hash
        - transactions: list of clinic contribution records
        - round_global_metrics: global model performance for that round

    This satisfies the thesis requirement for a transparent, auditable
    governance system (Section 5.3).
    """

    def __init__(self):
        self.chain: List[dict] = []
        self.pending_transactions: List[dict] = []
        self.token_balances: Dict[int, float] = {}
        self.valuator = ValuationEngine()
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = {
            "index":          0,
            "timestamp":      time.time(),
            "transactions":   [],
            "round_metrics":  {},
            "previous_hash":  "0" * 64,
            "proof":          0,
        }
        genesis["block_hash"] = self._hash_block(genesis)
        self.chain.append(genesis)

    @staticmethod
    def _hash_block(block: dict) -> str:
        block_copy = {k: v for k, v in block.items() if k != "block_hash"}
        block_str = json.dumps(block_copy, sort_keys=True, default=str)
        return hashlib.sha256(block_str.encode()).hexdigest()

    def log_round_contributions(self,
                                  round_num: int,
                                  global_metrics: dict,
                                  shapley_scores: Dict[int, float],
                                  clinic_data_info: Dict[int, dict],
                                  local_metrics_per_clinic: Dict[int, dict],
                                  drug_data_per_clinic: Dict[int, np.ndarray],
                                  verbose: bool = True) -> dict:
        """
        Called at the end of each federated round.
        Computes token awards for all clinics and seals a new block.
        """
        max_samples = max(v["n_train"] for v in clinic_data_info.values())
        round_transactions = []

        if verbose:
            print(f"\n{'='*65}")
            print(f"  📊 GOVERNANCE MODULE — ROUND {round_num} IP TOKEN ATTRIBUTION")
            print(f"{'='*65}")
            print(f"  Global Model AUC-ROC : {global_metrics.get('auc_roc', 0):.4f}")
            print(f"  Global Model Accuracy: {global_metrics.get('accuracy', 0):.4f}")
            print(f"\n  {'Clinic':<35} {'Shapley':>8} {'Vol':>6} {'Qual':>6} {'Tokens':>8}")
            print(f"  {'-'*65}")

        for cid in sorted(shapley_scores.keys()):
            info         = clinic_data_info[cid]
            local_met    = local_metrics_per_clinic.get(cid, {})
            drug_X       = drug_data_per_clinic.get(cid, np.zeros((1, 5)))
            shapley_val  = shapley_scores[cid]

            award = self.valuator.compute_token_award(
                clinic_id     = cid,
                shapley_score = shapley_val,
                n_samples     = info["n_train"],
                max_samples   = max_samples,
                local_metrics = local_met,
                drug_X        = drug_X,
            )
            award["round"] = round_num
            award["clinic_name"] = info.get("name", f"Clinic_{cid}")

            # Update balance
            self.token_balances[cid] = self.token_balances.get(cid, 0.0) + award["tokens_awarded"]
            round_transactions.append(award)

            if verbose:
                name_short = info.get("name", f"Clinic_{cid}")[:33]
                print(f"  {name_short:<35} {shapley_val:>+.4f} {award['volume_score']:>6.3f} "
                      f"{award['quality_score']:>6.3f} {award['tokens_awarded']:>7.2f} 🪙")

        self.pending_transactions = round_transactions

        if verbose:
            print(f"\n  {'💰 CUMULATIVE IP TOKEN BALANCES':}")
            print(f"  {'-'*45}")
            for cid in sorted(self.token_balances.keys()):
                name = clinic_data_info[cid].get("name", f"Clinic_{cid}")[:33]
                print(f"  {name:<35} {self.token_balances[cid]:>8.2f} 🪙")

        # Seal the block
        block = self._seal_block(round_num, global_metrics)

        if verbose:
            print(f"\n  🔗 Block #{block['index']} sealed")
            print(f"     Hash: {block['block_hash'][:16]}...{block['block_hash'][-8:]}")
            print(f"{'='*65}")

        return block

    def _seal_block(self, round_num: int, round_metrics: dict) -> dict:
        prev_hash = self.chain[-1]["block_hash"]
        block = {
            "index":         len(self.chain),
            "timestamp":     time.time(),
            "round":         round_num,
            "transactions":  self.pending_transactions.copy(),
            "round_metrics": round_metrics,
            "previous_hash": prev_hash,
            "proof":         round_num * 1000 + len(self.pending_transactions),
        }
        block["block_hash"] = self._hash_block(block)
        self.chain.append(block)
        self.pending_transactions = []
        return block

    def verify_chain_integrity(self) -> bool:
        """Verifies the hash chain has not been tampered with."""
        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]
            if current["previous_hash"] != previous["block_hash"]:
                return False
            recomputed = self._hash_block(
                {k: v for k, v in current.items() if k != "block_hash"}
            )
            if recomputed != current["block_hash"]:
                return False
        return True

    def get_ledger_summary(self) -> dict:
        """Returns a summary dict for thesis reporting."""
        return {
            "n_blocks":          len(self.chain),
            "n_transactions":    sum(len(b["transactions"]) for b in self.chain),
            "chain_intact":      self.verify_chain_integrity(),
            "token_balances":    dict(self.token_balances),
            "total_tokens_issued": sum(self.token_balances.values()),
        }

    def export_to_dataframe(self) -> "pd.DataFrame":
        """Exports all transactions to a pandas DataFrame for analysis."""
        import pandas as pd
        rows = []
        for block in self.chain[1:]:   # Skip genesis
            for tx in block["transactions"]:
                rows.append(tx)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PARTICIPATION INCENTIVE MULTIPLIER (cold-start bonus)
# ---------------------------------------------------------------------------

class ParticipationIncentiveManager:
    """
    Manages early-adopter and sustained-participation bonuses.

    Thesis motivation: Clinics that join early take on more risk (cold-start
    problem) and should be rewarded with a multiplier that decays over rounds.
    Also rewards clinics for not dropping out between rounds.
    """

    def __init__(self, n_rounds: int = 5):
        self.n_rounds = n_rounds
        self.participation_record: Dict[int, List[int]] = {}  # cid → rounds participated

    def record_participation(self, clinic_id: int, round_num: int):
        if clinic_id not in self.participation_record:
            self.participation_record[clinic_id] = []
        self.participation_record[clinic_id].append(round_num)

    def get_loyalty_multiplier(self, clinic_id: int, current_round: int) -> float:
        """
        Returns a loyalty multiplier (1.0 to 1.25).
        Clinics present in ALL rounds get +25% bonus on final tokens.
        """
        rounds = self.participation_record.get(clinic_id, [])
        max_possible = current_round
        if max_possible == 0:
            return 1.0
        participation_rate = len(rounds) / max_possible
        return 1.0 + 0.25 * participation_rate

    def get_early_adopter_bonus(self, clinic_id: int) -> float:
        """
        Clinics that joined in round 1 receive a one-time 15% bonus.
        """
        rounds = self.participation_record.get(clinic_id, [])
        if 1 in rounds:
            return 1.15
        return 1.0


if __name__ == "__main__":
    # Quick sanity test
    ledger = IPTokenLedger()
    print("Genesis block:", ledger.chain[0]["block_hash"][:20], "...")
    print("Chain integrity:", ledger.verify_chain_integrity())
    print("Governance module loaded successfully.")
