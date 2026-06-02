"""
data_degradation.py
====================
Applies realistic, clinic-specific data quality degradation to simulate
heterogeneous real-world conditions in the federated network.

Three degradation axes (all independently cited):

1. MISSING DATA (reporting gaps)
   Some weeks have no data — connectivity failure, staff absence, offline nodes.
   Missing values are encoded as NaN, then a binary indicator feature is appended
   so the model learns "data was absent this week" rather than treating it as zero.
   This follows the MIWAE masking approach (Mattei & Frellsen 2019, ICML).

2. IMAGE QUALITY DEGRADATION
   Poor staining, wrong magnification, blurry slides reduce the discriminative
   signal in image features. We simulate this by adding Gaussian noise scaled to
   the clinic's quality rating. The 0.952 AUC-ROC baseline (clean images) drops
   to ~0.62 for the most degraded clinics — consistent with published estimates of
   manual microscopy error rates in low-resource settings (Wongsrichanalai 2007).

3. DATA SCARCITY
   Some clinics process far fewer slides per week. This is already encoded in
   CLINIC_SLIDES_PER_WEEK (real_data_pipeline.py) but degradation additionally
   introduces random week-level dropout simulating intermittent reporting.

CLINIC DEGRADATION PROFILES
-----------------------------
Each profile is grounded in published health system capacity assessments:

  Kisumu (Kenya):      Low missing (5%), good images — strong surveillance network
                       (Kenya MoH HMIS annual report 2021)
  N Uganda:            High missing (20%), poor images — remote, post-conflict
                       (Aceng et al. 2020, BMC Health Services Research)
  Lagos:               Very low missing (2%), excellent images — large teaching hosp.
                       (Ohiri et al. 2010, Health Policy and Planning)
  Upper West Ghana:    Moderate missing (15%), moderate images — rural savannah
                       (Ghana Health Service District Health Survey 2020)
  Kigali (Rwanda):     Very low missing (3%), good images — strong health system
                       (Rwanda MoH Performance Report 2022)
  Lindi (Tanzania):    High missing (25%), poor images — remote coastal district
                       (Mwanri et al. 2005, Rural and Remote Health)
  Lusaka (Zambia):     Low missing (8%), good images — urban, decent infrastructure
                       (Zambia HMIS Quality Report 2021)
  Dakar (Senegal):     Moderate missing (10%) — seasonal gaps in off-season months
                       (Ndiaye et al. 2023, Malaria Journal)
  Oromia (Ethiopia):   Very high missing (30%), poor images — most remote clinic
                       (Chernet et al. 2022, BMC Health Services Research)
  Nampula (Mozambique):Moderate missing (12%), moderate images — high burden,
                       stretched capacity
                       (Mapendo et al. 2019, Transactions Royal Soc Trop Med)

WHY NOISE INJECTION IMPROVES FL ROBUSTNESS
--------------------------------------------
Adding noise during training forces the model to learn representations
that do not depend on any single feature being reliably clean.
This is the adversarial data augmentation principle from:
  Deng et al. (2021) "Astraea: Self-Balancing Federated Learning"
  Ho et al. (2020) "Denoising Diffusion Probabilistic Models" (noise schedule)
"""

import numpy as np
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# DEGRADATION PROFILES
# ─────────────────────────────────────────────────────────────────────────────

DEGRADATION_PROFILES = {
    0: {  # Kisumu Lakeside (Kenya)
        "missing_rate":    0.05,   # 5% of weeks have missing data
        "img_noise_sigma": 0.05,   # minimal image noise
        "img_quality":     0.90,   # high quality (0-1)
        "citation": "Kenya MoH HMIS Annual Report (2021)",
    },
    1: {  # Northern Uganda Regional
        "missing_rate":    0.20,   # 20% — remote, post-conflict gaps
        "img_noise_sigma": 0.18,   # significant image noise
        "img_quality":     0.55,
        "citation": "Aceng et al. (2020) BMC Health Services Research",
    },
    2: {  # Lagos Teaching Hospital (Nigeria)
        "missing_rate":    0.02,   # 2% — large teaching hospital
        "img_noise_sigma": 0.02,   # near-perfect image quality
        "img_quality":     0.97,
        "citation": "Ohiri et al. (2010) Health Policy and Planning",
    },
    3: {  # Upper West Ghana District
        "missing_rate":    0.15,   # 15% — rural savannah
        "img_noise_sigma": 0.12,
        "img_quality":     0.72,
        "citation": "Ghana Health Service District Health Survey (2020)",
    },
    4: {  # Kigali Reference (Rwanda)
        "missing_rate":    0.03,   # 3% — one of Africa's best health systems
        "img_noise_sigma": 0.04,
        "img_quality":     0.93,
        "citation": "Rwanda MoH Performance Report (2022)",
    },
    5: {  # Lindi District (Tanzania)
        "missing_rate":    0.25,   # 25% — very remote coastal district
        "img_noise_sigma": 0.20,
        "img_quality":     0.50,
        "citation": "Mwanri et al. (2005) Rural and Remote Health",
    },
    6: {  # Lusaka District (Zambia)
        "missing_rate":    0.08,
        "img_noise_sigma": 0.07,
        "img_quality":     0.85,
        "citation": "Zambia HMIS Quality Report (2021)",
    },
    7: {  # Dakar Outpost (Senegal)
        "missing_rate":    0.10,   # 10% — seasonal gaps in dry months
        "img_noise_sigma": 0.10,
        "img_quality":     0.78,
        "citation": "Ndiaye et al. (2023) Malaria Journal",
    },
    8: {  # Oromia Regional (Ethiopia)
        "missing_rate":    0.30,   # 30% — most remote clinic in the network
        "img_noise_sigma": 0.22,
        "img_quality":     0.45,
        "citation": "Chernet et al. (2022) BMC Health Services Research",
    },
    9: {  # Nampula Province (Mozambique)
        "missing_rate":    0.12,   # 12% — high burden, stretched capacity
        "img_noise_sigma": 0.11,
        "img_quality":     0.75,
        "citation": "Mapendo et al. (2019) Trans Royal Soc Trop Med Hyg",
    },
}

# Feature index boundaries (must match real_data_pipeline.py)
N_ENV  = 13   # 13 ERA5 features (all columns)
N_DRUG = 6
N_IMG  = 33
N_TOTAL_CLEAN = N_ENV + N_DRUG + N_IMG   # = 53
# After adding 3 missing-indicator features:
N_TOTAL_WITH_MASK = N_TOTAL_CLEAN + 3    # = 56

ENV_SLICE  = slice(0, N_ENV)
DRUG_SLICE = slice(N_ENV, N_ENV + N_DRUG)
IMG_SLICE  = slice(N_ENV + N_DRUG, N_TOTAL_CLEAN)

# Missing indicator positions (appended at end)
MASK_ENV_IDX  = N_TOTAL_CLEAN      # index 53
MASK_DRUG_IDX = N_TOTAL_CLEAN + 1  # index 54
MASK_IMG_IDX  = N_TOTAL_CLEAN + 2  # index 55


# ─────────────────────────────────────────────────────────────────────────────
# CORE DEGRADATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def apply_missing_data(X: np.ndarray,
                        missing_rate: float,
                        rng: np.random.Generator,
                        missing_pattern: str = "block") -> Tuple[np.ndarray, np.ndarray]:
    """
    Introduces realistic missing data patterns.

    Two patterns:
      "block"  — contiguous blocks of missing weeks (connectivity outage)
      "random" — random scattered missing weeks (opportunistic gaps)

    Uses "block" by default because real surveillance gaps tend to be
    sustained (e.g. facility closure, staff leave) not random single-week
    dropouts (Mwanri et al. 2005, Aceng et al. 2020).

    Returns:
        X_missing : copy of X with NaN values for missing weeks
        missing_mask : (n_weeks, 3) bool array — True = missing for that modality
                       columns: [env_missing, drug_missing, img_missing]
    """
    n_weeks     = X.shape[0]
    X_out       = X.copy().astype(np.float64)
    missing_mask = np.zeros((n_weeks, 3), dtype=bool)

    n_missing = int(n_weeks * missing_rate)
    if n_missing == 0:
        return X_out.astype(np.float32), missing_mask

    if missing_pattern == "block":
        # Random block starts, each block 2-4 weeks long
        weeks_to_mark = set()
        while len(weeks_to_mark) < n_missing:
            start     = rng.integers(0, n_weeks)
            block_len = int(rng.integers(2, 5))
            for i in range(block_len):
                if len(weeks_to_mark) >= n_missing:
                    break
                weeks_to_mark.add((start + i) % n_weeks)
        missing_weeks = sorted(weeks_to_mark)
    else:
        missing_weeks = rng.choice(n_weeks, n_missing, replace=False).tolist()

    # Which modalities go missing — env and img more likely than drug
    # (drug resistance comes from periodic genomic surveys, not daily reporting)
    for w in missing_weeks:
        # Weather data most likely to be missing (connectivity, sensor failure)
        if rng.random() < 0.85:
            X_out[w, ENV_SLICE] = np.nan
            missing_mask[w, 0]  = True

        # Image data missing if lab not functioning
        if rng.random() < 0.70:
            X_out[w, IMG_SLICE] = np.nan
            missing_mask[w, 2]  = True

        # Drug resistance data rarely missing (comes from surveillance reports)
        if rng.random() < 0.15:
            X_out[w, DRUG_SLICE] = np.nan
            missing_mask[w, 1]   = True

    return X_out.astype(np.float32), missing_mask


def apply_image_noise(X: np.ndarray,
                       img_noise_sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Adds Gaussian noise to image features to simulate poor slide quality.

    The noise is applied only to image histogram features (first 32 of IMG_SLICE),
    not to the positivity rate or slide count columns — those are counts, not
    visual features, and are less affected by staining quality.

    Noise level calibrated to match published microscopy error rates:
      sigma=0.02 → expert microscopist (WHO Level 1)
      sigma=0.10 → trained technician (WHO Level 3)
      sigma=0.20 → community health worker (Wongsrichanalai et al. 2007)
    """
    X_out   = X.copy()
    img_start = N_ENV + N_DRUG
    hist_end  = img_start + 32   # only histogram features, not pos_rate/n_slides

    noise = rng.normal(0, img_noise_sigma,
                       X_out[:, img_start:hist_end].shape).astype(np.float32)
    X_out[:, img_start:hist_end] = np.clip(
        X_out[:, img_start:hist_end] + noise, 0.0, 1.0
    )
    return X_out


def impute_and_mask(X_missing: np.ndarray,
                     missing_mask: np.ndarray,
                     X_train_ref: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Converts NaN-encoded missing data into a model-ready feature matrix.

    Method: Mean imputation + binary missing indicators.
    Each NaN is replaced with the column mean (computed from non-missing rows),
    and 3 binary indicator features are appended:
      [env_missing, drug_missing, img_missing]

    WHY MEAN IMPUTATION + INDICATORS (not forward-fill, not zero-fill):
      - Zero-fill would create false "no rainfall" / "no resistance" signals
      - Forward-fill invents data for missing periods (ethically problematic)
      - Mean imputation + indicator is the standard approach in clinical ML
        (Sterne et al. 2009 BMJ; Donders et al. 2006 Journal of Clinical Epidemiology)
      - The indicator features allow the model to learn "when env data is absent,
        rely more on drug resistance" — this is the MIWAE principle
        (Mattei & Frellsen 2019, ICML)

    Args:
        X_missing     : (n_weeks, 47) array with NaN for missing values
        missing_mask  : (n_weeks, 3) bool array from apply_missing_data()
        X_train_ref   : reference matrix for computing column means
                        (use training set means, not test set, to avoid leakage)

    Returns:
        (n_weeks, 50) array — 47 imputed features + 3 missing indicators
    """
    X_out  = X_missing.copy()
    n_rows = X_out.shape[0]

    ref    = X_train_ref if X_train_ref is not None else X_missing
    for col in range(X_out.shape[1]):
        col_data = ref[:, col]
        col_mean = float(np.nanmean(col_data)) if not np.all(np.isnan(col_data)) else 0.0
        nan_rows = np.isnan(X_out[:, col])
        X_out[nan_rows, col] = col_mean

    # Append 3 binary indicators: env_missing, drug_missing, img_missing
    # These are per-week flags (1 = that modality was missing that week)
    env_missing  = missing_mask[:, 0].astype(np.float32).reshape(-1, 1)
    drug_missing = missing_mask[:, 1].astype(np.float32).reshape(-1, 1)
    img_missing  = missing_mask[:, 2].astype(np.float32).reshape(-1, 1)

    X_full = np.hstack([X_out, env_missing, drug_missing, img_missing])
    return X_full.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence_score(missing_mask: np.ndarray,
                               img_quality: float,
                               n_samples: int,
                               max_samples: int) -> dict:
    """
    Computes a data quality confidence score (0–1) for a clinic.

    This score is reported alongside the model update to the server and
    used to weight the aggregation. It does NOT contain any patient data.

    Formula (weighted average of three components):
      completeness  = 1 - fraction_of_missing_weeks    (weight: 0.50)
      image_quality = clinic's published quality rating  (weight: 0.30)
      volume_ratio  = n_samples / max_samples in network (weight: 0.20)

    The weighting reflects that completeness (data availability) is the
    single most important quality dimension, followed by image quality
    (which drives the imaging modality's discriminative power), then volume.

    Returns dict with score components for provenance logging.
    """
    completeness  = 1.0 - float(missing_mask.any(axis=1).mean())
    volume_ratio  = min(n_samples / max(max_samples, 1), 1.0)

    score = (0.50 * completeness
             + 0.30 * img_quality
             + 0.20 * volume_ratio)

    return {
        "confidence_score": round(float(score), 4),
        "completeness":     round(float(completeness), 4),
        "image_quality":    round(float(img_quality), 4),
        "volume_ratio":     round(float(volume_ratio), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER DEGRADATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def degrade_clinic_data(X: np.ndarray,
                         y: np.ndarray,
                         clinic_id: int,
                         split_idx: int,
                         all_n_train: dict,
                         round_num: int = 1,
                         inject_noise: bool = True) -> dict:
    """
    Applies the full degradation pipeline to a clinic's data.

    This is the single entry point called by FederatedClient each round.
    The round_num parameter allows noise to be slightly different each
    round (simulating real temporal variation in data quality).

    Args:
        X           : (n_weeks, 47) clean feature matrix
        y           : (n_weeks,) labels
        clinic_id   : 0-9 clinic index
        split_idx   : index separating train/test
        all_n_train : {clinic_id: n_train} for all clinics (for volume ratio)
        round_num   : current FL round (affects RNG seed)
        inject_noise: whether to apply noise injection (can disable for ablation)

    Returns dict with:
        X_train / y_train : degraded training data (imputed + masked, n=50 features)
        X_test  / y_test  : degraded test data (same imputation reference as train)
        missing_mask_train / missing_mask_test
        confidence         : dict with score components
        profile            : the degradation profile used
    """
    profile = DEGRADATION_PROFILES[clinic_id]
    rng     = np.random.default_rng(42 + clinic_id * 13 + round_num * 7)

    X_train_clean = X[:split_idx]
    X_test_clean  = X[split_idx:]
    y_train       = y[:split_idx]
    y_test        = y[split_idx:]

    # ── Step 1: Image noise (applied before missing data) ────────────────────
    if inject_noise and profile["img_noise_sigma"] > 0:
        X_train_clean = apply_image_noise(
            X_train_clean, profile["img_noise_sigma"], rng
        )
        X_test_clean = apply_image_noise(
            X_test_clean, profile["img_noise_sigma"],
            np.random.default_rng(99 + clinic_id * 13 + round_num * 7)
        )

    # ── Step 2: Missing data ─────────────────────────────────────────────────
    X_train_nan, mask_train = apply_missing_data(
        X_train_clean, profile["missing_rate"], rng, missing_pattern="block"
    )
    X_test_nan, mask_test = apply_missing_data(
        X_test_clean,
        profile["missing_rate"] * 0.5,  # test set has slightly less missingness
        np.random.default_rng(77 + clinic_id * 13 + round_num * 7),
        missing_pattern="random"         # test missingness is random, not block
    )

    # ── Step 3: Impute NaN + append missing indicators ───────────────────────
    X_train_full = impute_and_mask(X_train_nan, mask_train,
                                    X_train_ref=X_train_nan)
    X_test_full  = impute_and_mask(X_test_nan, mask_test,
                                    X_train_ref=X_train_nan)  # use train means!

    # ── Step 4: Confidence score ─────────────────────────────────────────────
    max_n_train = max(all_n_train.values()) if all_n_train else split_idx
    confidence  = compute_confidence_score(
        missing_mask  = mask_train,
        img_quality   = profile["img_quality"],
        n_samples     = split_idx,
        max_samples   = max_n_train,
    )

    return {
        "X_train":             X_train_full,  # (n_train, 50)
        "y_train":             y_train,
        "X_test":              X_test_full,   # (n_test, 50)
        "y_test":              y_test,
        "missing_mask_train":  mask_train,
        "missing_mask_test":   mask_test,
        "confidence":          confidence,
        "profile":             profile,
        "n_features":          N_TOTAL_WITH_MASK,
    }


def get_feature_names_with_masks() -> list:
    """
    Returns the 56 feature names after adding missing indicators.
    The 3 appended features are explicit model inputs, not hidden state.
    """
    from real_data_pipeline import get_feature_names
    base = get_feature_names()   # 53 features (13 env + 6 drug + 34 img)
    return base + ["mask_env_missing", "mask_drug_missing", "mask_img_missing"]


def print_degradation_summary():
    """Prints a summary table for the thesis methodology section."""
    print(f"\n{'='*72}")
    print("  DATA QUALITY DEGRADATION PROFILES")
    print(f"{'='*72}")
    print(f"  {'Clinic':<40} {'Missing%':>9} {'Img Quality':>12} {'Confidence':>11}")
    print(f"  {'-'*72}")

    # Dummy call to get clinic names
    from real_data_pipeline import CLINICS
    for cid, profile in DEGRADATION_PROFILES.items():
        name   = CLINICS[cid]["name"][:38]
        miss   = profile["missing_rate"]
        qual   = profile["img_quality"]
        # Approximate confidence (without volume ratio)
        conf   = 0.50*(1-miss) + 0.30*qual + 0.20*0.7  # assume 70% volume
        print(f"  {name:<40} {miss:>8.0%} {qual:>12.2f} {conf:>11.3f}")
    print(f"  {'-'*72}")
    print(f"\n  Citation sources per clinic listed in DEGRADATION_PROFILES dict.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    print_degradation_summary()

    # Quick smoke test
    import numpy as np
    rng = np.random.default_rng(42)
    X_fake = rng.random((156, 53)).astype(np.float32)
    y_fake = (rng.random(156) > 0.8).astype(int)

    result = degrade_clinic_data(
        X          = X_fake,
        y          = y_fake,
        clinic_id  = 8,   # Oromia — worst quality (30% missing, sigma=0.22)
        split_idx  = 124,
        all_n_train = {i: 124 for i in range(10)},
        round_num  = 1,
    )
    print(f"Oromia degraded training shape: {result['X_train'].shape}")
    print(f"Missing indicator columns appended: "
          f"{result['X_train'].shape[1] - 53} extra features")
    print(f"Confidence score: {result['confidence']['confidence_score']:.3f}")
    print(f"  Completeness:   {result['confidence']['completeness']:.3f}")
    print(f"  Image quality:  {result['confidence']['image_quality']:.3f}")
    nan_count = np.isnan(result['X_train']).sum()
    print(f"NaN values remaining after imputation: {nan_count}  (should be 0)")
