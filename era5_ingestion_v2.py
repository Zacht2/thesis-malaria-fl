"""
era5_ingestion_v2.py
====================
Handles your specific ERA5 folder structure:

  era5_hydro_daily/
      tile1_hydro_2024_01.nc   (ZIP-wrapped, contains tp + e)
      tile2_hydro_2024_01.nc
      ...

  era5_land_daily/
      tile1/
          temp_daily_mean_2024_07/
              2m_temperature_0_daily-mean.nc
              2m_dewpoint_temperature_0_daily-mean.nc
              skin_temperature_0_daily-mean.nc
          temp_daily_minimum_2025_01/
              2m_temperature_0_daily-min.nc
              ...
          temp_daily_maximum_2025_02/
              2m_temperature_0_daily-max.nc
              ...
          veg_daily_mean_2025_12/
              leaf_area_index_high_vegetation_0_daily-mean.nc
              leaf_area_index_low_vegetation_0_daily-mean.nc
          pressure_daily_mean_2025_07/
              data.nc
          hydro_daily_mean_2024_01/
              ...
      tile2/ ...
      tile3/ ...
      tile4/ ...

USAGE:
    python3 era5_ingestion_v2.py \
        --hydro_dir ~/Downloads/era5_hydro_daily \
        --land_dir  ~/Downloads/era5_land_daily \
        --output_dir ~/Desktop/era5_weekly

REQUIREMENTS:
    pip install xarray netCDF4 numpy pandas
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile, tempfile, os, argparse, warnings
warnings.filterwarnings("ignore")

# ── Clinic coordinates ───────────────────────────────────────────────────────
CLINICS = {
    "Kisumu_KEN":        {"lat":  -0.102, "lon":  34.762},
    "N.Uganda_UGA":      {"lat":   2.778, "lon":  32.299},
    "Lagos_NGA":         {"lat":   6.524, "lon":   3.379},
    "Upper_W.Ghana_GHA": {"lat":  10.059, "lon":  -2.508},
    "Kigali_RWA":        {"lat":  -1.944, "lon":  30.060},
    "Lindi_TZA":         {"lat": -10.000, "lon":  39.714},
    "Lusaka_ZMB":        {"lat": -15.417, "lon":  28.283},
    "Dakar_SEN":         {"lat":  14.693, "lon": -17.447},
    "Oromia_ETH":        {"lat":   7.550, "lon":  40.000},
    "Nampula_MOZ":       {"lat": -15.116, "lon":  39.267},
}

# ── How to recognise variables from their filename or xarray name ────────────
# (filename fragment → our label,  aggregation to use)
FILENAME_HINTS = {
    "2m_temperature":                    ("t2m",   "mean"),
    "2m_dewpoint":                       ("d2m",   "mean"),
    "skin_temperature":                  ("skt",   "mean"),
    "leaf_area_index_high_vegetation":   ("lai_hv","mean"),
    "leaf_area_index_low_vegetation":    ("lai_lv","mean"),
}

# xarray variable short names → our label
# Includes alternative long-form CF names ERA5 sometimes uses
VAR_HINTS = {
    # Precipitation
    "tp":                                        ("precip_m",  "sum"),
    # Evaporation — ERA5 uses short "e" OR long CF name
    "e":                                         ("evap_m",    "sum"),
    "lwe_thickness_of_water_evaporation_amount": ("evap_m",    "sum"),
    "evaporation":                               ("evap_m",    "sum"),
    # Temperature
    "t2m":                                       ("t2m",       "mean"),
    "2m_temperature":                            ("t2m",       "mean"),
    # Dewpoint
    "d2m":                                       ("d2m",       "mean"),
    "2m_dewpoint_temperature":                   ("d2m",       "mean"),
    # Skin temperature — sometimes stored as "skt" or "skin_temperature"
    "skt":                                       ("skt",       "mean"),
    "skin_temperature":                          ("skt",       "mean"),
    # Other
    "sp":                                        ("sp",        "mean"),
    "surface_pressure":                          ("sp",        "mean"),
    "swvl1":                                     ("swvl1",     "mean"),
    "volumetric_soil_water_layer_1":             ("swvl1",     "mean"),
    "lai_hv":                                    ("lai_hv",    "mean"),
    "leaf_area_index_high_vegetation":           ("lai_hv",    "mean"),
    "lai_lv":                                    ("lai_lv",    "mean"),
    "leaf_area_index_low_vegetation":            ("lai_lv",    "mean"),
}

# folder name fragments → override aggregation
FOLDER_AGG = {
    "minimum": "min",
    "maximum": "max",
    "mean":    "mean",
    "sum":     "sum",
}

def open_nc(path: Path) -> xr.Dataset:
    """Open a NetCDF file whether plain HDF5 or ZIP-wrapped."""
    with open(path, 'rb') as f:
        magic = f.read(4)
    if magic[:2] == b'PK':
        with zipfile.ZipFile(path, 'r') as z:
            inner = [n for n in z.namelist() if n.endswith('.nc')]
            if not inner:
                raise ValueError(f"No .nc inside zip {path.name}")
            tmp = tempfile.mktemp(suffix='.nc')
            with z.open(inner[0]) as src, open(tmp, 'wb') as dst:
                dst.write(src.read())
        ds = xr.open_dataset(tmp, engine='netcdf4')
        ds.load()          # read into RAM before deleting temp file
        os.unlink(tmp)
        return ds
    return xr.open_dataset(path, engine='netcdf4')

def guess_var_and_agg(path: Path, ds: xr.Dataset):
    """
    Returns (our_label, xr_varname, agg) by checking filename hints first,
    then xarray variable names. Returns None if unrecognised.
    """
    fname = path.stem.lower()
    parent = path.parent.name.lower()

    # Work out aggregation from folder name
    agg = "mean"
    for frag, a in FOLDER_AGG.items():
        if frag in parent:
            agg = a
            break

    # Try filename hints
    for hint, (label, default_agg) in FILENAME_HINTS.items():
        if hint.lower() in fname:
            # Find the matching xarray variable
            for var in ds.data_vars:
                if hint.split('_')[0] in var.lower() or var.lower() in hint.lower():
                    return label, var, agg
            # If exact var not found, just take first data var
            if ds.data_vars:
                return label, list(ds.data_vars)[0], agg

    # Try xarray variable name hints
    for var in ds.data_vars:
        if var in VAR_HINTS:
            label, default_agg = VAR_HINTS[var]
            return label, var, agg

    return None  # unrecognised

def extract_point_daily(ds: xr.Dataset, var: str,
                         lat: float, lon: float) -> pd.Series:
    """Extract nearest point as a DAILY series. Resampling to weekly
    happens later, after all monthly files are concatenated, so that
    weeks spanning month boundaries are never dropped."""
    lat_dim = next((c for c in ds.coords if c in ('lat','latitude')), None)
    lon_dim = next((c for c in ds.coords if c in ('lon','longitude')), None)
    time_dim = next((c for c in ds.coords
                     if 'time' in c or 'valid' in c), None)

    if lat_dim is None or lon_dim is None:
        raise ValueError(f"No lat/lon coords: {list(ds.coords)}")

    da = ds[var].sel({lat_dim: lat, lon_dim: lon}, method='nearest')

    if time_dim and time_dim != 'time':
        da = da.rename({time_dim: 'time'})

    s = da.to_series()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def daily_to_weekly(s: pd.Series, agg: str) -> pd.Series:
    """Resample a continuous daily series to Monday-anchored weeks."""
    resample_map = {
        'sum':  lambda x: x.resample('W-MON', label='left', closed='left').sum(),
        'mean': lambda x: x.resample('W-MON', label='left', closed='left').mean(),
        'min':  lambda x: x.resample('W-MON', label='left', closed='left').min(),
        'max':  lambda x: x.resample('W-MON', label='left', closed='left').max(),
    }
    return resample_map[agg](s)

def dewpoint_to_rh(t_c: pd.Series, d_c: pd.Series) -> pd.Series:
    """Magnus formula → relative humidity %"""
    rh = 100 * (np.exp(17.625 * d_c / (243.04 + d_c)) /
                np.exp(17.625 * t_c / (243.04 + t_c)))
    return rh.clip(0, 100)

def build_clinic_csv(clinic: str, coords: dict,
                     series: dict, out_dir: Path,
                     start_date: str = None, end_date: str = None):
    """
    series: dict of our_label → pd.Series (weekly)
    Merges everything into one CSV with all pipeline features.
    """
    idx = None
    for s in series.values():
        if idx is None or len(s) > len(idx):
            idx = s.index

    df = pd.DataFrame(index=idx)
    df.index.name = 'week_start'

    # Precipitation (m → mm)
    if 'precip_m' in series:
        df['precip_mm'] = series['precip_m'].reindex(idx) * 1000
        for lag, weeks in [(7,1),(14,2),(21,3),(28,4)]:
            df[f'precip_lag{lag}_mm'] = df['precip_mm'].shift(weeks)

    # Evaporation (m → mm, ERA5 evap is negative → abs)
    if 'evap_m' in series:
        df['evaporation_mm'] = series['evap_m'].reindex(idx).abs() * 1000

    # Temperature (K → C)
    if 't2m' in series:
        df['temp_mean_c'] = series['t2m'].reindex(idx) - 273.15

    # Use daily min/max if available, else approximate from mean
    # We collect t2m separately for min and max folders
    if 't2m_min' in series:
        df['temp_min_c'] = series['t2m_min'].reindex(idx) - 273.15
    elif 'temp_mean_c' in df.columns:
        df['temp_min_c'] = df['temp_mean_c'] - 3.0

    if 't2m_max' in series:
        df['temp_max_c'] = series['t2m_max'].reindex(idx) - 273.15
    elif 'temp_mean_c' in df.columns:
        df['temp_max_c'] = df['temp_mean_c'] + 3.0

    # Humidity from dewpoint + temp
    t_for_rh = series.get('t2m', series.get('t2m_mean'))
    d_for_rh = series.get('d2m', series.get('d2m_mean'))
    if t_for_rh is not None and d_for_rh is not None:
        t_c = t_for_rh.reindex(idx) - 273.15
        d_c = d_for_rh.reindex(idx) - 273.15
        df['humidity_pct'] = dewpoint_to_rh(t_c, d_c)
        df['humidity_14d_pct'] = df['humidity_pct'].rolling(2, min_periods=1).mean()

    # Skin temperature
    if 'skt' in series:
        df['skin_temp_c'] = series['skt'].reindex(idx) - 273.15

    # Surface pressure
    if 'sp' in series:
        df['surface_pressure_pa'] = series['sp'].reindex(idx)

    # Soil water
    if 'swvl1' in series:
        df['soil_water_m3m3'] = series['swvl1'].reindex(idx)

    # LAI
    if 'lai_hv' in series:
        df['lai_hv'] = series['lai_hv'].reindex(idx)
    if 'lai_lv' in series:
        df['lai_lv'] = series['lai_lv'].reindex(idx)

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    out_path = out_dir / f"era5_{clinic}.csv"
    df.to_csv(out_path)
    n_cols = df.notna().any().sum()
    print(f"  ✓ {clinic}: {len(df)} weeks, {n_cols}/{len(df.columns)} populated columns → {out_path.name}")
    return df

def process_all(hydro_dir: Path, land_dir: Path, out_dir: Path,
                start_date: str = None, end_date: str = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all nc files
    all_files = []
    if hydro_dir.exists():
        all_files += sorted(hydro_dir.rglob('*.nc'))
    if land_dir.exists():
        all_files += sorted(land_dir.rglob('*.nc'))

    print(f"\nFound {len(all_files)} .nc files total")

    # For each clinic, accumulate weekly series by label
    # clinic_name → {label → [Series, Series, ...]}
    clinic_series: dict[str, dict[str, list]] = {c: {} for c in CLINICS}

    for i, fpath in enumerate(all_files):
        try:
            ds = open_nc(fpath)
        except Exception as e:
            print(f"  ⚠ Could not open {fpath.name}: {e}")
            continue

        result = guess_var_and_agg(fpath, ds)
        if result is None:
            ds.close()
            continue

        label, var, agg = result

        # Refine label for min/max folders so we keep them separate
        parent = fpath.parent.name.lower()
        if 'minimum' in parent:
            label = label + '_min'
        elif 'maximum' in parent:
            label = label + '_max'
        elif 'mean' in parent:
            label = label + '_mean' if label in ('t2m','d2m','skt') else label

        # Extract daily values for each clinic (weekly resampling happens later)
        for clinic, coords in CLINICS.items():
            try:
                s = extract_point_daily(ds, var, coords['lat'], coords['lon'])
                key = (label, agg)
                if key not in clinic_series[clinic]:
                    clinic_series[clinic][key] = []
                clinic_series[clinic][key].append(s)
            except Exception:
                pass

        ds.close()

        if (i+1) % 20 == 0:
            print(f"  ... processed {i+1}/{len(all_files)} files")

    print(f"\nBuilding clinic CSVs...")
    for clinic, coords in CLINICS.items():
        # Concatenate all daily chunks FIRST, then resample to weekly once
        # This prevents gaps at month boundaries where a week spans two files
        merged = {}
        for (label, agg), series_list in clinic_series[clinic].items():
            daily = pd.concat(series_list).sort_index()
            daily = daily[~daily.index.duplicated(keep='first')]
            weekly = daily_to_weekly(daily, agg)
            merged[label] = weekly
        build_clinic_csv(clinic, coords, merged, out_dir, start_date, end_date)

    print(f"\n✅ Done. CSVs saved to {out_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hydro_dir',  default='~/Downloads/era5_hydro_daily',
                   help='Path to era5_hydro_daily folder')
    p.add_argument('--land_dir',   default='~/Downloads/era5_land_daily',
                   help='Path to era5_land_daily folder')
    p.add_argument('--output_dir', default='~/Desktop/era5_weekly',
                   help='Where to save the 10 clinic CSVs')
    p.add_argument('--start_date', default='2024-01-01')
    p.add_argument('--end_date',   default='2026-03-01')
    args = p.parse_args()

    process_all(
        hydro_dir  = Path(args.hydro_dir).expanduser(),
        land_dir   = Path(args.land_dir).expanduser(),
        out_dir    = Path(args.output_dir).expanduser(),
        start_date = args.start_date,
        end_date   = args.end_date,
    )

if __name__ == '__main__':
    main()
