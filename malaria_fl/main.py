"""
main.py
=======
Entry point for the Malaria Federated Learning Framework.

Usage:
    python main.py                          # Synthetic images (no image dir needed)
    python main.py --images ./cell_images  # Use real Kaggle images
    python main.py --rounds 10             # More FL rounds
    python main.py --no-cache              # Regenerate all data fresh

Expected Kaggle image folder structure:
    cell_images/
        Parasitized/   (13,779 PNG files)
        Uninfected/    (13,779 PNG files)
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Malaria FL Framework — Federated Outbreak Prediction"
    )
    parser.add_argument("--images",   type=str, default=None,
                        help="Path to Kaggle cell images folder (Parasitized/ + Uninfected/)")
    parser.add_argument("--era5",     type=str, default=None,
                        help="Path to ERA5 weekly CSV folder produced by era5_ingestion_v2.py")
    parser.add_argument("--pf8",      type=str, default=None,
                        help="Path to MalariaGEN Pf8-samples.csv")
    parser.add_argument("--map",      type=str, default=None,
                        help="Path to MAP Subnational_Unit-data.csv")
    parser.add_argument("--rounds",   type=int, default=5,
                        help="Number of federated learning rounds (default: 5)")
    parser.add_argument("--data-dir", type=str, default="clinic_data",
                        help="Directory for cached clinic data (default: clinic_data)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Regenerate all synthetic data (ignore cache)")
    parser.add_argument("--quiet",    action="store_true",
                        help="Reduce verbosity")
    return parser.parse_args()


def check_image_dir(image_dir: str) -> bool:
    """Validates the Kaggle image directory structure."""
    if image_dir is None:
        return False
    from pathlib import Path
    para_dir  = Path(image_dir) / "Parasitized"
    uninf_dir = Path(image_dir) / "Uninfected"
    if not para_dir.exists() or not uninf_dir.exists():
        print(f"  ⚠️  Image dir '{image_dir}' found but missing Parasitized/ or Uninfected/ subfolder.")
        print(f"     Falling back to synthetic image features.")
        return False
    n_para  = len(list(para_dir.glob("*.png")))
    n_uninf = len(list(uninf_dir.glob("*.png")))
    print(f"  ✅ Real images found: {n_para} Parasitized + {n_uninf} Uninfected")
    return True


def clear_cache(data_dir: str):
    """Removes cached clinic data to force regeneration."""
    from pathlib import Path
    cache_files = list(Path(data_dir).glob("clinic_*.pkl"))
    for f in cache_files:
        f.unlink()
    print(f"  🗑️  Cleared {len(cache_files)} cached files.")


def main():
    args = parse_args()

    print("\n" + "╔" + "═"*63 + "╗")
    print("║  MALARIA FEDERATED LEARNING FRAMEWORK                        ║")
    print("║  Proactive Surveillance for Sub-Saharan Africa               ║")
    print("║  Bachelor's Thesis — Zachary Thurston, ESADE 2025-26         ║")
    print("╚" + "═"*63 + "╝")

    # Clear cache if requested
    if args.no_cache and os.path.exists(args.data_dir):
        clear_cache(args.data_dir)

    # Validate image directory
    use_real_images = check_image_dir(args.images)
    image_dir = args.images if use_real_images else None

    if not use_real_images:
        print("\n  ℹ️  Running with SYNTHETIC image features.")
        print("     To use your Kaggle images, run:")
        print("     python main.py --images /path/to/cell_images")

    era5_dir = args.era5
    if era5_dir:
        print(f"\n  🌦️  ERA5 real weather: {era5_dir}")
    else:
        print("\n  ℹ️  Running with SYNTHETIC weather (WorldClim calibrated).")
        print("     To use real ERA5, run:")
        print("     python main.py --era5 ~/Desktop/era5_weekly")

    # Auto-locate Pf8 and MAP files: check --pf8/--map args first,
    # then look next to the script, then fall back to a clear error.
    from pathlib import Path
    script_dir = Path(__file__).parent.resolve()

    def _find_file(arg_val, candidates, label):
        if arg_val:
            p = Path(arg_val).expanduser()
            if p.exists():
                return str(p)
            print(f"  ⚠️  {label} not found at: {p}")
        for name in candidates:
            for search in [script_dir, Path.home()/"Desktop"/"malaria_fl",
                           Path.home()/"Desktop", Path.home()/"Downloads"]:
                p = search / name
                if p.exists():
                    print(f"  📂 {label} auto-detected: {p}")
                    return str(p)
        return None

    pf8_path = _find_file(args.pf8, ["Pf8-samples.csv"], "Pf8 file")
    map_path = _find_file(args.map, ["Subnational_Unit-data.csv"], "MAP file")

    if not pf8_path:
        print("\n  ❌ Cannot find Pf8-samples.csv")
        print("     Download it from the uploads sidebar and place it in your malaria_fl folder,")
        print("     or pass: --pf8 /path/to/Pf8-samples.csv")
        sys.exit(1)
    if not map_path:
        print("\n  ❌ Cannot find Subnational_Unit-data.csv")
        print("     Download it from the uploads sidebar and place it in your malaria_fl folder,")
        print("     or pass: --map /path/to/Subnational_Unit-data.csv")
        sys.exit(1)

    print(f"  📊 Pf8:  {pf8_path}")
    print(f"  🗺️  MAP:  {map_path}")

    # Import here so errors surface cleanly
    from federated_engine import FederatedSimulation

    t_start = time.time()

    sim = FederatedSimulation(
        image_dir = image_dir,
        era5_dir  = era5_dir,
        pf8_path  = pf8_path,
        map_path  = map_path,
        data_dir  = args.data_dir,
        n_rounds  = args.rounds,
        verbose   = not args.quiet,
    )

    sim.run()

    # Save outputs
    results_path = sim.save_results("fl_results.pkl")
    ledger_df    = sim.export_governance_csv("ip_token_ledger.csv")

    elapsed = time.time() - t_start
    print(f"\n  ⏱️  Total runtime: {elapsed:.1f}s")
    print(f"  📁 Outputs:")
    print(f"     fl_results.pkl      — full simulation results")
    print(f"     ip_token_ledger.csv — IP attribution ledger")
    print(f"     clinic_data/        — cached clinic datasets")

    # Print a quick ledger preview for thesis
    print(f"\n  📋 IP TOKEN LEDGER PREVIEW (first 5 rows):")
    if not ledger_df.empty:
        cols = ["round", "clinic_name", "tokens_awarded", "shapley_score",
                "volume_score", "quality_score", "rarity_score"]
        available_cols = [c for c in cols if c in ledger_df.columns]
        print(ledger_df[available_cols].head(5).to_string(index=False))

    print("\n  ✅ Done.\n")


if __name__ == "__main__":
    main()
