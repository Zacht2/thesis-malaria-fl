"""
era5_pipeline_adapter.py
========================
Drop-in replacement for the synthetic weather section of real_data_pipeline.py.

HOW TO USE:
    1. Run era5_ingestion.py to produce era5_weekly/ CSV files
    2. In real_data_pipeline.py, replace the call to _build_synthetic_weather()
       with:

           from era5_pipeline_adapter import ERA5WeatherAdapter
           adapter = ERA5WeatherAdapter(era5_dir='./era5_weekly')
           env_features = adapter.get_env_features(clinic_name, week_index)

    3. The returned numpy array has the same shape as the existing env_features
       so nothing else in the pipeline needs to change.

FEATURE ORDER (must match real_data_pipeline.py env_feature_cols):
    [0]  precip_mm
    [1]  precip_lag7_mm
    [2]  precip_lag14_mm
    [3]  precip_lag21_mm
    [4]  precip_lag28_mm
    [5]  temp_min_c
    [6]  temp_max_c
    [7]  humidity_14d_pct
    [8]  skin_temp_c           (new — was zero in synthetic)
    [9]  soil_water_m3m3       (new — was zero in synthetic)
    [10] lai_hv                (new — was zero in synthetic)
    [11] lai_lv                (new — was zero in synthetic)
    [12] evaporation_mm        (new — was zero in synthetic)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Same column ordering the GBT was trained on (env block, 13 features)
ENV_COLS = [
    'precip_mm', 'precip_lag7_mm', 'precip_lag14_mm',
    'precip_lag21_mm', 'precip_lag28_mm',
    'temp_min_c', 'temp_max_c', 'humidity_14d_pct',
    'skin_temp_c', 'soil_water_m3m3', 'lai_hv', 'lai_lv', 'evaporation_mm'
]

class ERA5WeatherAdapter:
    """
    Loads pre-processed ERA5 weekly CSVs and serves env feature vectors
    indexed by clinic and week number.
    """

    def __init__(self, era5_dir: str = './era5_weekly'):
        self.era5_dir = Path(era5_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def _load(self, clinic_name: str) -> pd.DataFrame:
        """Load and cache the ERA5 CSV for a clinic."""
        # Normalise clinic name to match file naming
        safe_name = clinic_name.replace(' ', '_').replace('/', '_')
        csv_path = self.era5_dir / f"era5_{safe_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"ERA5 CSV not found for clinic '{clinic_name}': {csv_path}\n"
                f"Run era5_ingestion.py first, or check the clinic name."
            )
        df = pd.read_csv(csv_path, index_col='week_start', parse_dates=True)
        # Add any missing columns as NaN (graceful fallback)
        for col in ENV_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df[ENV_COLS]

    def _get_df(self, clinic_name: str) -> pd.DataFrame:
        if clinic_name not in self._cache:
            self._cache[clinic_name] = self._load(clinic_name)
        return self._cache[clinic_name]

    def get_env_features(self, clinic_name: str,
                          week_index: int,
                          fill_missing: bool = True) -> np.ndarray:
        """
        Return a 1-D numpy array of shape (len(ENV_COLS),) for one week.

        Parameters
        ----------
        clinic_name : str   e.g. 'Kisumu_KEN'
        week_index  : int   0-based row index into the weekly time series
        fill_missing: bool  if True, forward-fill then zero-fill NaN values

        Returns
        -------
        np.ndarray  shape (13,), dtype float32
        """
        df = self._get_df(clinic_name)
        if week_index >= len(df):
            raise IndexError(
                f"week_index {week_index} out of range for {clinic_name} "
                f"(n={len(df)} weeks)"
            )
        row = df.iloc[week_index].copy()
        if fill_missing:
            # Try forward fill from previous row
            if week_index > 0:
                prev = df.iloc[week_index - 1]
                row = row.fillna(prev)
            row = row.fillna(0.0)
        return row.values.astype(np.float32)

    def get_env_matrix(self, clinic_name: str) -> np.ndarray:
        """
        Return the full weekly feature matrix for a clinic.
        Shape: (n_weeks, 13), dtype float32.
        """
        df = self._get_df(clinic_name).ffill().fillna(0.0)
        return df.values.astype(np.float32)

    def available_weeks(self, clinic_name: str) -> pd.DatetimeIndex:
        """Return the datetime index of available weeks for a clinic."""
        return self._get_df(clinic_name).index

    def coverage_report(self) -> pd.DataFrame:
        """Print a summary of data availability across all clinics."""
        rows = []
        for csv in sorted(self.era5_dir.glob('era5_*.csv')):
            clinic = csv.stem.replace('era5_', '')
            df = pd.read_csv(csv, index_col='week_start', parse_dates=True)
            completeness = df.notna().mean().mean() * 100
            rows.append({
                'clinic': clinic,
                'weeks': len(df),
                'start': str(df.index[0].date()),
                'end':   str(df.index[-1].date()),
                'completeness_%': round(completeness, 1),
                'features': list(df.columns)
            })
        return pd.DataFrame(rows)


# ── Convenience function for direct use in real_data_pipeline.py ────────────

_adapter_singleton: ERA5WeatherAdapter | None = None

def init_era5(era5_dir: str = './era5_weekly'):
    """Call once at pipeline startup."""
    global _adapter_singleton
    _adapter_singleton = ERA5WeatherAdapter(era5_dir)
    report = _adapter_singleton.coverage_report()
    print("\nERA5 coverage report:")
    print(report.to_string(index=False))
    return _adapter_singleton

def get_era5_features(clinic_name: str, week_index: int) -> np.ndarray:
    """Stateless convenience wrapper — call init_era5() first."""
    if _adapter_singleton is None:
        raise RuntimeError("Call init_era5(era5_dir) before get_era5_features()")
    return _adapter_singleton.get_env_features(clinic_name, week_index)
