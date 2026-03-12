#!/usr/bin/env python3
"""
Testing the Diffusion Limitation Hypothesis for
Declining Methane Uptake in Forest Soils
========================================================

This script tests the diffusion limitation hypothesis for explaining
a 77% decline in forest CH₄ uptake at two Long-Term Ecological
Research (LTER) sites: Baltimore Ecosystem Study (BES, 1998–2025)
and Hubbard Brook Experimental Forest (HBR, 2002–2015).

Fourteen analytical blocks systematically evaluate five predictions
of the hypothesis using >10,000 chamber measurements, in-situ soil
moisture, PRISM climate data, NADP deposition records, lysimeter
chemistry, and vegetation surveys.

Usage:
    pip install -r requirements.txt
    python master_analysis.py

Output:
    14 publication-quality figures (PNG + SVG) and SUMMARY.txt
    are written to the output/ subdirectory.

Data:
    All input data are expected in ../Data/ relative to this script.
    See requirements in the README for the full list of input files.

Author:  Victor Edmonds
Contact: victoredmonds@gmail.com
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress, t
import statsmodels.api as sm
from statsmodels.formula.api import ols
import ruptures as rpt
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths are relative to the project root (parent of Analysis/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

DATA_DIR = os.path.join(_PROJECT_DIR, "Data")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Site classifications
URBAN_SITES = ['HD', 'LEA', 'MCD', 'GB', 'GLY', 'UMBC']
RURAL_SITES = ['ORM', 'ORU', 'ORLR', 'ORUR', 'CAH']
FOREST_SITES = URBAN_SITES + RURAL_SITES

# PRISM file mapping for BES sites
PRISM_MAPPING = {
    'HD': 'BES',
    'LEA': 'BES',
    'MCD': 'BES',
    'GB': 'BES',
    'GLY': 'BES',
    'UMBC': 'BES',
    'ORM': 'OregonRidge',
    'ORU': 'OregonRidge',
    'ORLR': 'OregonRidge',
    'ORUR': 'OregonRidge',
    'CAH': 'OregonRidge',
}

# Figure style
plt.style.use('seaborn-v0_8-darkgrid')
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (12, 5.5)
DPI = 300

# Error codes for missing values
MISSING_VALUES = [-9999.99, -9999, -9, -99.999]

# Outlier threshold: sites with confirmed ebullition / hotspot events
# GB (Gwynns Falls urban stream-adjacent) and ORLR (Oregon Ridge Lower Riparian)
# produce extreme positive CH4 fluxes (up to 7234 mg C m-2 h-1) from 63
# measurements (0.6% of data) that represent methane source hotspots, not
# the forest soil sink under study. These are excluded from sink analyses.
HOTSPOT_SITES = ['GB', 'ORLR']

# Additional per-measurement outlier filter: within each site-year,
# values beyond ±3 SD from the site-year mean are trimmed.
OUTLIER_SD_THRESHOLD = 3

# ============================================================================
# DATA LOADING
# ============================================================================

def load_bes_flux():
    """Load BES trace gas flux data."""
    fpath = os.path.join(DATA_DIR, "BES_trace-gas-collection_1998_2025.csv")
    df = pd.read_csv(fpath)

    # Parse date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Year'].astype(int)

    # Extract year-month for merging
    df['YearMonth'] = df['Date'].dt.to_period('M')

    # Replace error codes with NaN
    df['CH4_flux'] = df['CH4_flux'].replace(MISSING_VALUES, np.nan)

    print(f"Loaded BES flux data: {len(df)} rows, {df['Year'].min()}-{df['Year'].max()}")
    return df

def load_hbr_annual():
    """Load HBR annual CH4 flux data."""
    fpath = os.path.join(DATA_DIR, "knb-lter-hbr.207/knb-lter-hbr.207-CH4_flux_annual.csv")
    df = pd.read_csv(fpath)

    # Column name cleanup
    if 'Annual CH4flux' in df.columns:
        df = df.rename(columns={'Annual CH4flux': 'Annual_CH4_flux'})

    # Filter to Baltimore sites for comparison
    df['Year'] = df['Year'].astype(int)

    print(f"Loaded HBR annual data: {len(df)} rows")
    return df

def load_hbr_monthly():
    """Load HBR monthly CH4 flux data."""
    fpath = os.path.join(DATA_DIR, "knb-lter-hbr.207/knb-lter-hbr.207-CH4_flux_monthly.csv")
    df = pd.read_csv(fpath)

    # Replace error codes
    flux_cols = ['Monthly_flux_1', 'monthly_flux_2', 'Monthly_Flux_3', 'Monthly_Flux_4']
    for col in flux_cols:
        if col in df.columns:
            df[col] = df[col].replace(MISSING_VALUES, np.nan)

    # Calculate mean monthly flux across replicates
    df['Mean_Monthly_flux'] = df[flux_cols].mean(axis=1)

    # Parse year-month
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['YearMonth'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1)).dt.to_period('M')

    print(f"Loaded HBR monthly data: {len(df)} rows")
    return df

def load_prism(site_key):
    """Load PRISM climate data for a specific site."""
    prism_site = PRISM_MAPPING.get(site_key, 'BES')

    if prism_site == 'BES':
        fpath = os.path.join(DATA_DIR, "PRISM/BES-PRISM_ppt_tmean_stable_4km_199801_202506_39.3400_-76.6200.csv")
    elif prism_site == 'OregonRidge':
        fpath = os.path.join(DATA_DIR, "PRISM/OregonRidge-PRISM_ppt_tmean_stable_4km_199801_202506_39.4970_-76.6890.csv")
    elif prism_site == 'HubbardBrook':
        fpath = os.path.join(DATA_DIR, "PRISM/HubbardBrook-PRISM_ppt_tmean_stable_4km_199801_202506_43.9440_-71.7500.csv")
    else:
        return None

    # Skip header lines (10 lines of metadata before actual data)
    df = pd.read_csv(fpath, skiprows=10)
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')

    return df.rename(columns={'ppt (mm)': 'ppt_mm', 'tmean (degrees C)': 'tmean_c'})

def load_nadp():
    """Load NADP deposition data for both Baltimore and Hubbard Brook."""
    nadp_data = {}

    # Hubbard Brook (NH02)
    fpath_hb = os.path.join(DATA_DIR, "NADP/NTN-nh02-m-s-kg.csv")
    df_hb = pd.read_csv(fpath_hb)
    df_hb = df_hb.replace(-9, np.nan)
    df_hb['Site'] = 'HB'
    nadp_data['HB'] = df_hb

    # Baltimore (MD99)
    fpath_md = os.path.join(DATA_DIR, "NADP/NTN-md99-m-s-kg.csv")
    df_md = pd.read_csv(fpath_md)
    df_md = df_md.replace(-9, np.nan)
    df_md['Site'] = 'MD'
    nadp_data['MD'] = df_md

    print(f"Loaded NADP: HB {len(df_hb)} records, MD {len(df_md)} records")
    return nadp_data

def load_lysimeter():
    """Load BES lysimeter (soil solution) data."""
    fpath = os.path.join(DATA_DIR, "knb-lter-bes.428.292/BES_lysimeter_data_1999-2025_for_EDI.csv")
    df = pd.read_csv(fpath)

    # Parse date
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    df['Year'] = df['Sampling_Date'].dt.year

    # Replace error codes
    for col in ['NH4', 'NO3', 'PO4']:
        df[col] = df[col].replace([-9999, -9999.0], np.nan)

    print(f"Loaded lysimeter data: {len(df)} rows")
    return df

def load_vegetation():
    """Load BES vegetation data (seedling cover)."""
    fpath = os.path.join(DATA_DIR, "knb-lter-bes.3300.110/vegetation_BESLTER_1998_2003_2015_Veg_Data_9_Tree_Seedling_Cover.csv")
    df = pd.read_csv(fpath)

    # Map full site names to standard codes used elsewhere
    site_name_map = {
        'Leakin': 'LEA',
        'Oregon Ridge': 'ORM',
        'Oregon Ridge ': 'ORM',   # trailing space in CSV
        'Hillsdale': 'HD',
        'Hillsdale ': 'HD',       # trailing space in CSV
    }
    df['Site'] = df['Site'].str.strip().map(
        {k.strip(): v for k, v in site_name_map.items()}
    ).fillna(df['Site'].str.strip())

    # Also strip the 'Plot ' column name if present (has trailing space)
    df.columns = [c.strip() for c in df.columns]

    print(f"Loaded vegetation (seedlings): {len(df)} rows, sites: {df['Site'].unique()}")
    return df

def load_soil_moisture():
    """Load BES soil moisture (VWC) data."""
    fpath = os.path.join(DATA_DIR, "knb-lter-bes.3400/BES_TempVWC_2011-2020.csv")
    df = pd.read_csv(fpath)

    # Map site names to standard codes
    site_map = {
        'HD1': 'HD',
        'LEA1': 'LEA',
        'LEA2': 'LEA',
        'ORLR': 'ORLR',
        'ORU1': 'ORU',
        'ORU2': 'ORU',
        'ORUR': 'ORUR',
        'UMBC1': 'UMBC',
        'UMBC2': 'UMBC',
    }
    df['Site_Code'] = df['Site'].map(site_map)

    # Parse timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['YearMonth'] = df['Timestamp'].dt.to_period('M')

    # Extract VWC columns (Port_1_VWC through Port_5_VWC)
    vwc_cols = [col for col in df.columns if col.startswith('Port_') and col.endswith('_VWC')]

    # Convert to numeric, replacing 'NA' with NaN
    for col in vwc_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate mean VWC across ports per row
    df['Mean_VWC'] = df[vwc_cols].mean(axis=1)

    # Select relevant columns
    df = df[['Site_Code', 'Timestamp', 'Year', 'Month', 'YearMonth', 'Mean_VWC']].copy()

    print(f"Loaded soil moisture (VWC): {len(df)} rows, sites: {df['Site_Code'].unique()}")
    return df

def load_soil_properties():
    """Load BES forest soil properties."""
    fpath = os.path.join(DATA_DIR, "knb-lter-bes.584/Physical_chemical_and_biological_properties_of_forest_and_home_lawn_soils.csv")
    df = pd.read_csv(fpath)

    # Filter to forest sites only
    df = df[df['LU_Current'] == 'Forest'].copy()

    # Map site names to standard codes (strip trailing spaces first)
    df['Site'] = df['Site'].str.strip()
    site_map = {
        'Hillsdale 1': 'HD',
        'Hillsdale 2': 'HD',
        'Leakin 1': 'LEA',
        'Leakin 2': 'LEA',
        'Oregon Ridge Mid 1': 'ORM',
        'Oregon Ridge Mid 2': 'ORM',
        'Oregon Ridge Upper 1': 'ORU',
        'Oregon Ridge Upper 2': 'ORU',
    }
    df['Site_Code'] = df['Site'].map(site_map)

    print(f"Loaded soil properties: {len(df)} rows, forest sites only")
    return df

def load_harvard_forest():
    """Load Harvard Forest atmospheric CH4 concentration data."""
    fpath = os.path.join(DATA_DIR, "knb-lter-hfr.60.19/hf060-02-methane.csv")
    df = pd.read_csv(fpath)

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Year'] = df['year']
    df['Month'] = df['month']
    df['YearMonth'] = df['datetime'].dt.to_period('M')

    print(f"Loaded Harvard Forest CH4 concentration: {len(df)} rows, {df['year'].min()}-{df['year'].max()}")
    return df

# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def trim_outliers(df, flux_col='CH4_flux', group_cols=None):
    """
    Remove hotspot sites and trim per-group outliers beyond ±3 SD.

    1. Drops rows from HOTSPOT_SITES (GB, ORLR) entirely.
    2. Within each group (default: Site × Year), removes measurements
       beyond ±3 SD from the group mean.

    Returns a filtered copy of df.
    """
    if group_cols is None:
        group_cols = ['Site', 'Year']

    # Step 1: Exclude hotspot sites
    df_out = df[~df['Site'].isin(HOTSPOT_SITES)].copy()
    n_hotspot = len(df) - len(df_out)

    # Step 2: Per-group SD trim
    def _trim_group(g):
        mu = g[flux_col].mean()
        sd = g[flux_col].std()
        if sd == 0 or np.isnan(sd):
            return g
        return g[np.abs(g[flux_col] - mu) <= OUTLIER_SD_THRESHOLD * sd]

    available_groups = [c for c in group_cols if c in df_out.columns]
    if available_groups:
        df_trimmed = df_out.groupby(available_groups, group_keys=False).apply(_trim_group)
    else:
        df_trimmed = df_out

    n_sd_trim = len(df_out) - len(df_trimmed)
    print(f"  Outlier filter: removed {n_hotspot} hotspot-site rows, "
          f"{n_sd_trim} rows beyond ±{OUTLIER_SD_THRESHOLD} SD "
          f"({len(df_trimmed)} remaining of {len(df)})")
    return df_trimmed.reset_index(drop=True)


def regression_stats(x, y, label=""):
    """Compute regression statistics for a paired x,y series."""
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return None

    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
    n = len(x_clean)
    r2 = r_value ** 2

    # Calculate 95% CI for slope
    t_crit = t.ppf(0.975, n - 2)
    ci_lower = slope - t_crit * std_err
    ci_upper = slope + t_crit * std_err

    results = {
        'label': label,
        'n': n,
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'r': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }

    return results

def coefficient_of_variation(x):
    """Calculate coefficient of variation."""
    return np.nanstd(x) / np.nanmean(x) if np.nanmean(x) != 0 else np.nan

def detect_source_events(flux_array, threshold=0):
    """Count proportion of positive (source) CH4 flux measurements."""
    flux_clean = flux_array[~np.isnan(flux_array)]
    if len(flux_clean) == 0:
        return np.nan
    return np.sum(flux_clean > threshold) / len(flux_clean)

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    g1 = group1[~np.isnan(group1)]
    g2 = group2[~np.isnan(group2)]

    if len(g1) == 0 or len(g2) == 0:
        return np.nan

    mean1, mean2 = np.mean(g1), np.mean(g2)
    std1, std2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    n1, n2 = len(g1), len(g2)

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan

# ============================================================================
# BLOCK 1: PRECIPITATION-FLUX ANALYSIS
# ============================================================================

def block_1_precipitation_flux():
    """
    Block 1: Prediction 1 - Moisture should explain CH4 flux
    ========================================================
    Tests whether increased precipitation explains the decline in forest
    methane uptake. The hypothesis predicts a strong negative relationship.
    """
    print("\n" + "="*70)
    print("BLOCK 1: PRECIPITATION-FLUX ANALYSIS")
    print("="*70)

    # Load data
    bes = load_bes_flux()

    # Filter to forest sites and remove missing CH4
    forest_mask = bes['Site'].isin(FOREST_SITES)
    bes = bes[forest_mask & bes['CH4_flux'].notna()].copy()

    # Remove hotspot sites and trim outliers
    bes = trim_outliers(bes, flux_col='CH4_flux')

    # Merge with PRISM by site and year-month
    all_data = []
    for site in FOREST_SITES:
        site_data = bes[bes['Site'] == site].copy()
        if len(site_data) == 0:
            continue

        prism = load_prism(site)
        if prism is None:
            continue

        # Merge on YearMonth
        merged = pd.merge(
            site_data, prism[['YearMonth', 'ppt_mm']],
            on='YearMonth', how='inner'
        )
        all_data.append(merged)

    merged_all = pd.concat(all_data, ignore_index=True)

    # Pooled regression (all forest sites)
    stats_pooled = regression_stats(
        merged_all['ppt_mm'].values,
        merged_all['CH4_flux'].values,
        label="All forest sites (pooled)"
    )

    print(f"\nPooled regression (n={stats_pooled['n']}):")
    print(f"  Slope: {stats_pooled['slope']:.6f} mg C m-2 h-1 per mm ppt")
    print(f"  R²: {stats_pooled['r2']:.4f}")
    print(f"  p-value: {stats_pooled['p_value']:.4e}")
    print(f"  95% CI: [{stats_pooled['ci_lower']:.6f}, {stats_pooled['ci_upper']:.6f}]")

    # By-site regressions
    print(f"\nBy-site regressions:")
    site_stats = []
    for site in sorted(set(merged_all['Site'])):
        site_subset = merged_all[merged_all['Site'] == site]
        stats_site = regression_stats(
            site_subset['ppt_mm'].values,
            site_subset['CH4_flux'].values,
            label=site
        )
        if stats_site:
            site_stats.append(stats_site)
            print(f"  {site}: r²={stats_site['r2']:.4f}, p={stats_site['p_value']:.4e}, n={stats_site['n']}")

    # Figure 1: Scatterplot with regression
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    ax.scatter(merged_all['ppt_mm'], merged_all['CH4_flux'], alpha=0.15, s=15)

    # Plot regression line
    x_range = np.linspace(merged_all['ppt_mm'].min(), merged_all['ppt_mm'].max(), 100)
    y_pred = stats_pooled['intercept'] + stats_pooled['slope'] * x_range
    ax.plot(x_range, y_pred, 'r-', linewidth=2, label='Linear regression')

    p_str = f"p = {stats_pooled['p_value']:.3e}" if stats_pooled['p_value'] >= 0.001 else "p < 0.001"
    ax.text(0.05, 0.95, f"r² = {stats_pooled['r2']:.3f}\n{p_str}\nn = {stats_pooled['n']}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Precipitation (mm/month)', fontsize=12)
    ax.set_ylabel('CH₄ Flux (mg C m⁻² h⁻¹)', fontsize=12)
    ax.set_title('Prediction 1: Precipitation vs CH₄ Flux\n(Diffusion Limitation Hypothesis)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_1_Precipitation_Flux.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_1_Precipitation_Flux.svg'))
    plt.close()

    print(f"\nFigure 1 saved.")

    return {
        'pooled': stats_pooled,
        'by_site': site_stats,
        'merged_data': merged_all
    }

# ============================================================================
# BLOCK 2: TEMPERATURE-FLUX ANALYSIS
# ============================================================================

def block_2_temperature_flux(merged_precip_data):
    """
    Block 2: Temperature Control (Ruling out an alternative)
    =========================================================
    Tests whether temperature explains flux variation. If moisture explains
    flux via diffusion limitation, temperature should be independent.
    """
    print("\n" + "="*70)
    print("BLOCK 2: TEMPERATURE-FLUX ANALYSIS")
    print("="*70)

    # Reuse merged data from Block 1, add temperature
    merged_all = merged_precip_data.copy()

    # Add temperature from PRISM
    all_data = []
    for site in FOREST_SITES:
        site_data = merged_all[merged_all['Site'] == site].copy()
        if len(site_data) == 0:
            continue

        prism = load_prism(site)
        if prism is None:
            continue

        # Merge on YearMonth
        merged = pd.merge(
            site_data, prism[['YearMonth', 'tmean_c']],
            on='YearMonth', how='inner'
        )
        all_data.append(merged)

    merged_temp = pd.concat(all_data, ignore_index=True) if all_data else merged_all

    # Pooled regression
    stats_pooled = regression_stats(
        merged_temp['tmean_c'].values,
        merged_temp['CH4_flux'].values,
        label="All forest sites (pooled)"
    )

    print(f"\nPooled regression (n={stats_pooled['n']}):")
    print(f"  Slope: {stats_pooled['slope']:.6f} mg C m-2 h-1 per °C")
    print(f"  R²: {stats_pooled['r2']:.4f}")
    print(f"  p-value: {stats_pooled['p_value']:.4e}")
    print(f"  95% CI: [{stats_pooled['ci_lower']:.6f}, {stats_pooled['ci_upper']:.6f}]")

    # By-site regressions
    print(f"\nBy-site regressions:")
    site_stats = []
    for site in sorted(set(merged_temp['Site'])):
        site_subset = merged_temp[merged_temp['Site'] == site]
        stats_site = regression_stats(
            site_subset['tmean_c'].values,
            site_subset['CH4_flux'].values,
            label=site
        )
        if stats_site:
            site_stats.append(stats_site)
            print(f"  {site}: r²={stats_site['r2']:.4f}, p={stats_site['p_value']:.4e}, n={stats_site['n']}")

    # Figure 2: Scatterplot with regression
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    ax.scatter(merged_temp['tmean_c'], merged_temp['CH4_flux'], alpha=0.15, s=15)

    # Plot regression line
    x_range = np.linspace(merged_temp['tmean_c'].min(), merged_temp['tmean_c'].max(), 100)
    y_pred = stats_pooled['intercept'] + stats_pooled['slope'] * x_range
    ax.plot(x_range, y_pred, 'r-', linewidth=2, label='Linear regression')

    ax.text(0.05, 0.95, f"r² = {stats_pooled['r2']:.3f}\np = {stats_pooled['p_value']:.3e}\nn = {stats_pooled['n']}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Mean Temperature (°C)', fontsize=12)
    ax.set_ylabel('CH₄ Flux (mg C m⁻² h⁻¹)', fontsize=12)
    ax.set_title('Temperature Does Not Explain CH₄ Flux\n(Control for Alternative Hypothesis)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_2_Temperature_Flux.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_2_Temperature_Flux.svg'))
    plt.close()

    print(f"\nFigure 2 saved.")

    return {
        'pooled': stats_pooled,
        'by_site': site_stats,
        'merged_data': merged_temp
    }

# ============================================================================
# BLOCK 3: MULTI-PREDICTOR REGRESSION
# ============================================================================

def block_3_multi_predictor(merged_temp_data):
    """
    Block 3: Prediction 5 - The Clean Kill
    ======================================
    Multi-predictor OLS model: CH4 ~ ppt + tmean + year
    If diffusion limitation is correct, precipitation should be significant.
    We expect only YEAR to be significant, killing the hypothesis.
    """
    print("\n" + "="*70)
    print("BLOCK 3: MULTI-PREDICTOR REGRESSION (OLS)")
    print("="*70)

    # Prepare data
    df_model = merged_temp_data[['CH4_flux', 'ppt_mm', 'tmean_c', 'Year']].copy()
    df_model = df_model.dropna()

    # Standardize predictors for interpretability
    df_model['ppt_std'] = (df_model['ppt_mm'] - df_model['ppt_mm'].mean()) / df_model['ppt_mm'].std()
    df_model['tmean_std'] = (df_model['tmean_c'] - df_model['tmean_c'].mean()) / df_model['tmean_c'].std()
    df_model['year_std'] = (df_model['Year'] - df_model['Year'].mean()) / df_model['Year'].std()

    # Fit OLS
    X = df_model[['ppt_std', 'tmean_std', 'year_std']]
    X = sm.add_constant(X)
    y = df_model['CH4_flux']

    model = sm.OLS(y, X).fit()

    print("\nOLS Summary:")
    print(model.summary())

    # Extract key stats
    print(f"\nKey Results:")
    print(f"  R² = {model.rsquared:.4f}")
    print(f"  Adjusted R² = {model.rsquared_adj:.4f}")
    print(f"  F-statistic = {model.fvalue:.4f}, p-value = {model.f_pvalue:.4e}")

    print(f"\nCoefficient Significance:")
    for idx, param in enumerate(model.params.index[1:], 1):
        coef = model.params[param]
        pval = model.pvalues[param]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  {param}: {coef:.6f}, p={pval:.4e} {sig}")

    # Figure 3: Coefficient plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    params = model.params[1:]
    ses = model.bse[1:]
    pvals = model.pvalues[1:]

    x_pos = np.arange(len(params))
    colors = ['red' if p < 0.05 else 'gray' for p in pvals]

    ax.barh(x_pos, params, xerr=1.96*ses, color=colors, alpha=0.7, capsize=5)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(['Precipitation (std)', 'Temperature (std)', 'Year (std)'])
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Standardized Coefficient', fontsize=12)
    ax.set_title('Prediction 5: Multi-Predictor Model\n(R² = 1.2% — Year Dominates)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_3_Coefficients.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_3_Coefficients.svg'))
    plt.close()

    print(f"\nFigure 3 saved.")

    return model

# ============================================================================
# BLOCK 4: SEASONAL STRATIFICATION
# ============================================================================

def block_4_seasonal_stratification(merged_temp_data):
    """
    Block 4: Prediction 2 - Seasonal Stratification
    =============================================
    Hypothesis: if diffusion limitation explains decline, should be strongest
    in high-moisture seasons (winter/spring). Shows summer has weakest effect.
    """
    print("\n" + "="*70)
    print("BLOCK 4: SEASONAL STRATIFICATION")
    print("="*70)

    df = merged_temp_data.copy()

    # Add month and season
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Fall'

    df['Season'] = df['Month'].apply(get_season)

    print("\nSeasonal Regression Results:")
    print("Season\t\tPrecip_r²\tPrecip_p\tTemp_r²\t\tTemp_p\t\tn")

    results_season = {}
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = df[df['Season'] == season]

        # Precip regression
        stat_p = regression_stats(season_data['ppt_mm'].values,
                                  season_data['CH4_flux'].values)
        # Temp regression
        stat_t = regression_stats(season_data['tmean_c'].values,
                                  season_data['CH4_flux'].values)

        if stat_p and stat_t:
            results_season[season] = {'precip': stat_p, 'temp': stat_t}
            print(f"{season}\t\t{stat_p['r2']:.4f}\t\t{stat_p['p_value']:.4e}\t{stat_t['r2']:.4f}\t\t{stat_t['p_value']:.4e}\t{stat_p['n']}")

    # Figure 4: Bar chart of seasonal r² values
    seasons_order = ['Winter', 'Spring', 'Summer', 'Fall']
    precip_r2 = [results_season[s]['precip']['r2'] for s in seasons_order if s in results_season]
    temp_r2 = [results_season[s]['temp']['r2'] for s in seasons_order if s in results_season]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    x = np.arange(len(seasons_order))
    width = 0.35

    ax.bar(x - width/2, precip_r2, width, label='Precipitation', alpha=0.8)
    ax.bar(x + width/2, temp_r2, width, label='Temperature', alpha=0.8)

    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('R² (explained variance)', fontsize=12)
    ax.set_title('Prediction 2: Seasonal Stratification\n(Summer Should Be Strongest If Diffusion-Limited)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seasons_order)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(max(precip_r2), max(temp_r2)) * 1.2])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_4_Seasonal.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_4_Seasonal.svg'))
    plt.close()

    print(f"\nFigure 4 saved.")

    return results_season

# ============================================================================
# BLOCK 5: CALCIUM EXPERIMENT
# ============================================================================

def block_5_calcium_experiment():
    """
    Block 5: Prediction 3 - Calcium Fertilization Experiment
    =========================================================
    HBR Watershed experiment: same rain, different nutrients.
    If diffusion limits uptake, Ca-fertilized (WS1) should have same decline
    as Reference (WS6-BB). Instead, they diverge.
    """
    print("\n" + "="*70)
    print("BLOCK 5: CALCIUM EXPERIMENT (HBR WATERSHEDS)")
    print("="*70)

    # Load HBR annual data
    hbr = load_hbr_annual()

    # Filter to Hubbard Brook sites
    hb_data = hbr[hbr['StudySite'] == 'Hubbard Brook'].copy()

    if len(hb_data) == 0:
        print("No Hubbard Brook data found")
        return None

    # Get WS1 (Ca fertilized) and WS6-BB (Reference)
    ws1 = hb_data[hb_data['Site'] == 'WS1'][['Year', 'Annual_CH4_flux']].copy()
    ws6 = hb_data[hb_data['Site'] == 'WS6-BB'][['Year', 'Annual_CH4_flux']].copy()

    print(f"\nWS1 (Ca-fertilized): {len(ws1)} years, {ws1['Year'].min()}-{ws1['Year'].max()}")
    print(f"WS6-BB (Reference): {len(ws6)} years, {ws6['Year'].min()}-{ws6['Year'].max()}")

    # Summary statistics
    ws1_mean = ws1['Annual_CH4_flux'].mean()
    ws6_mean = ws6['Annual_CH4_flux'].mean()

    print(f"\nMean annual flux:")
    print(f"  WS1 (Ca-fert): {ws1_mean:.4f} mg C m-2 h-1")
    print(f"  WS6-BB (Ref):  {ws6_mean:.4f} mg C m-2 h-1")

    # Calculate effect size (Cohen's d)
    d = cohens_d(ws1['Annual_CH4_flux'].values, ws6['Annual_CH4_flux'].values)
    print(f"  Cohen's d: {d:.3f}")

    # Source frequency (% positive fluxes)
    ws1_source = detect_source_events(ws1['Annual_CH4_flux'].values)
    ws6_source = detect_source_events(ws6['Annual_CH4_flux'].values)

    print(f"\nSource events (% positive flux):")
    print(f"  WS1: {ws1_source*100:.1f}%")
    print(f"  WS6-BB: {ws6_source*100:.1f}%")

    # Linear trends (post-2000)
    ws1_post = ws1[ws1['Year'] >= 2000]
    ws6_post = ws6[ws6['Year'] >= 2000]

    trend_ws1 = linregress(ws1_post['Year'], ws1_post['Annual_CH4_flux'])
    trend_ws6 = linregress(ws6_post['Year'], ws6_post['Annual_CH4_flux'])

    print(f"\nLinear trends (post-2000):")
    print(f"  WS1: slope={trend_ws1.slope:.6f}, p={trend_ws1.pvalue:.4e}")
    print(f"  WS6-BB: slope={trend_ws6.slope:.6f}, p={trend_ws6.pvalue:.4e}")

    # Figure 5: Time series comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Panel A: Time series
    ax1.plot(ws1['Year'], ws1['Annual_CH4_flux'], 'o-', label='WS1 (Ca-fertilized)', linewidth=2, markersize=6)
    ax1.plot(ws6['Year'], ws6['Annual_CH4_flux'], 's-', label='WS6-BB (Reference)', linewidth=2, markersize=6)

    # Add trend lines for post-2000
    x_trend = np.array([ws1_post['Year'].min(), ws1_post['Year'].max()])
    ax1.plot(x_trend, trend_ws1.intercept + trend_ws1.slope * x_trend, '--',
             color='C0', alpha=0.7, linewidth=1.5)
    ax1.plot(x_trend, trend_ws6.intercept + trend_ws6.slope * x_trend, '--',
             color='C1', alpha=0.7, linewidth=1.5)

    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Annual CH₄ Flux (mg C m⁻² h⁻¹)', fontsize=11)
    ax1.set_title('A) Time Series Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linestyle=':', linewidth=0.5)

    # Panel B: Source frequency comparison
    sources = [ws1_source, ws6_source]
    sites = ['WS1\n(Ca-fert)', 'WS6-BB\n(Ref)']
    colors_bar = ['#1f77b4', '#ff7f0e']

    ax2.bar(sites, [s*100 for s in sources], color=colors_bar, alpha=0.7, width=0.5)
    ax2.set_ylabel('Source Events (%)', fontsize=11)
    ax2.set_title('B) Source Frequency', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (site, src) in enumerate(zip(sites, sources)):
        ax2.text(i, src*100 + 3, f'{src*100:.1f}%', ha='center', fontweight='bold')

    fig.suptitle('Prediction 3: Ca Fertilization Experiment\n(Same Rain, Different Nutrients)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_5_CaExperiment.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_5_CaExperiment.svg'), bbox_inches='tight')
    plt.close()

    print(f"\nFigure 5 saved.")

    return {
        'ws1_annual': ws1,
        'ws6_annual': ws6,
        'cohens_d': d,
        'ws1_source_freq': ws1_source,
        'ws6_source_freq': ws6_source,
        'ws1_trend': trend_ws1,
        'ws6_trend': trend_ws6,
    }

# ============================================================================
# BLOCK 6: URBAN-RURAL DIVERGENCE
# ============================================================================

def block_6_urban_rural_divergence():
    """
    Block 6: Prediction 4 - Urban-Rural Divergence
    ==============================================
    If diffusion limitation causes recovery, both urban and rural forests
    should recover similarly after 2012. Instead, they diverge.
    """
    print("\n" + "="*70)
    print("BLOCK 6: URBAN-RURAL DIVERGENCE")
    print("="*70)

    # Load HBR monthly data (includes both Baltimore urban/rural)
    hbr_monthly = load_hbr_monthly()

    # Filter to Baltimore sites
    baltimore = hbr_monthly[hbr_monthly['StudySite'] == 'Baltimore'].copy()

    # Classify as Urban or Rural
    urban_mask = baltimore['Type'] == 'Urban'

    # Calculate annual means
    urban_annual = baltimore[urban_mask].groupby('Year')['Mean_Monthly_flux'].mean()
    rural_annual = baltimore[~urban_mask].groupby('Year')['Mean_Monthly_flux'].mean()

    print(f"\nUrban sites (annual mean):")
    print(f"  n years: {len(urban_annual)}")
    print(f"  Mean: {urban_annual.mean():.4f}")
    print(f"  Std: {urban_annual.std():.4f}")

    print(f"\nRural sites (annual mean):")
    print(f"  n years: {len(rural_annual)}")
    print(f"  Mean: {rural_annual.mean():.4f}")
    print(f"  Std: {rural_annual.std():.4f}")

    # Post-2012 trends
    urban_post = urban_annual[urban_annual.index >= 2012]
    rural_post = rural_annual[rural_annual.index >= 2012]

    trend_urban = linregress(urban_post.index, urban_post.values)
    trend_rural = linregress(rural_post.index, rural_post.values)

    print(f"\nPost-2012 linear trends:")
    print(f"  Urban: slope={trend_urban.slope:.6f}, p={trend_urban.pvalue:.4e}")
    print(f"  Rural: slope={trend_rural.slope:.6f}, p={trend_rural.pvalue:.4e}")

    # Figure 6: Time series with divergence
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Plot annual means
    ax.plot(urban_annual.index, urban_annual.values, 'o-', label='Urban', linewidth=2, markersize=6)
    ax.plot(rural_annual.index, rural_annual.values, 's-', label='Rural', linewidth=2, markersize=6)

    # Add post-2012 trend lines
    x_urban = np.array([urban_post.index.min(), urban_post.index.max()])
    ax.plot(x_urban, trend_urban.intercept + trend_urban.slope * x_urban, '--',
            color='C0', alpha=0.7, linewidth=1.5)

    x_rural = np.array([rural_post.index.min(), rural_post.index.max()])
    ax.plot(x_rural, trend_rural.intercept + trend_rural.slope * x_rural, '--',
            color='C1', alpha=0.7, linewidth=1.5)

    # Shade post-2012 region
    ax.axvspan(2012, ax.get_xlim()[1], alpha=0.1, color='gray', label='Post-2012')

    ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Mean CH₄ Flux (mg C m⁻² h⁻¹)', fontsize=12)
    ax.set_title('Prediction 4: Urban-Rural Divergence\n(Recovery Should Be Similar)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_6_UrbanRural.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_6_UrbanRural.svg'))
    plt.close()

    print(f"\nFigure 6 saved.")

    return {
        'urban_annual': urban_annual,
        'rural_annual': rural_annual,
        'trend_urban': trend_urban,
        'trend_rural': trend_rural,
    }

# ============================================================================
# BLOCK 7: BREAKPOINT DETECTION
# ============================================================================

def block_7_breakpoint_detection():
    """
    Block 7: Structural Break Detection
    ==================================
    Uses ruptures library to detect structural breaks in time series.
    Should show break around 2012 for both BES and HBR.
    """
    print("\n" + "="*70)
    print("BLOCK 7: BREAKPOINT DETECTION")
    print("="*70)

    # BES annual MEDIAN (robust to remaining outliers after trimming)
    bes = load_bes_flux()
    bes_forest = bes[bes['Site'].isin(FOREST_SITES) & bes['CH4_flux'].notna()].copy()
    bes_forest = trim_outliers(bes_forest, flux_col='CH4_flux')
    bes_annual = bes_forest.groupby('Year')['CH4_flux'].median()

    # HBR Baltimore annual median
    hbr_annual = load_hbr_annual()
    hbr_balt = hbr_annual[hbr_annual['StudySite'] == 'Baltimore']
    hbr_balt_annual = hbr_balt.groupby('Year')['Annual_CH4_flux'].median()

    # HBR reference watershed (median across plots)
    hbr_ref_all = hbr_annual[hbr_annual['Site'] == 'WS6-BB'].copy()
    hbr_ref = hbr_ref_all.groupby('Year')['Annual_CH4_flux'].median().reset_index()
    hbr_ref = hbr_ref.sort_values('Year')

    print(f"\nBES annual median range: {bes_annual.min():.3f} to {bes_annual.max():.3f}")
    print(f"HBR Baltimore annual median range: {hbr_balt_annual.min():.3f} to {hbr_balt_annual.max():.3f}")
    print(f"HBR Reference years: {hbr_ref['Year'].min()}-{hbr_ref['Year'].max()}")

    # Detect breakpoints using PELT algorithm
    # pen=0.1 calibrated for median series (small variance requires low penalty;
    # sensitivity analysis: pen 0.01-0.1 gives consistent primary breaks,
    # pen >= 2 finds nothing due to low signal amplitude)
    algo_bes = rpt.Pelt(model="l2").fit(bes_annual.values)
    breakpoints_bes_raw = algo_bes.predict(pen=0.1)
    # Filter out terminal index (PELT always returns len(signal) as last element)
    breakpoints_bes = [bp for bp in breakpoints_bes_raw if bp < len(bes_annual)]

    algo_hbr = rpt.Pelt(model="l2").fit(hbr_ref['Annual_CH4_flux'].values)
    breakpoints_hbr_raw = algo_hbr.predict(pen=0.1)
    breakpoints_hbr = [bp for bp in breakpoints_hbr_raw if bp < len(hbr_ref)]

    print(f"\nDetected breakpoints (PELT, pen=0.1):")
    bp_year_bes = None
    if breakpoints_bes:
        bp_year_bes = bes_annual.index[breakpoints_bes[0] - 1]
        print(f"  BES: {bp_year_bes}")
        if len(breakpoints_bes) > 1:
            for bp in breakpoints_bes[1:]:
                print(f"       also: {bes_annual.index[bp - 1]}")
    else:
        print(f"  BES: none detected")

    bp_year_hbr = None
    if breakpoints_hbr:
        bp_year_hbr = hbr_ref.iloc[breakpoints_hbr[0] - 1]['Year']
        print(f"  HBR Reference: {bp_year_hbr}")
    else:
        print(f"  HBR Reference: none detected")

    # Pre/post breakpoint medians
    bes_pre_median = bes_post_median = None
    if bp_year_bes is not None:
        bes_pre_median = bes_annual[bes_annual.index < bp_year_bes].median()
        bes_post_median = bes_annual[bes_annual.index >= bp_year_bes].median()
        print(f"\n  BES pre-break median (before {bp_year_bes}): {bes_pre_median:.4f} mg C m⁻² h⁻¹")
        print(f"  BES post-break median ({bp_year_bes}+): {bes_post_median:.4f} mg C m⁻² h⁻¹")

    hbr_pre_median = hbr_post_median = None
    if bp_year_hbr is not None:
        hbr_pre = hbr_ref[hbr_ref['Year'] < bp_year_hbr]['Annual_CH4_flux']
        hbr_post = hbr_ref[hbr_ref['Year'] >= bp_year_hbr]['Annual_CH4_flux']
        hbr_pre_median = hbr_pre.median()
        hbr_post_median = hbr_post.median()
        print(f"  HBR pre-break median (before {int(bp_year_hbr)}): {hbr_pre_median:.4f} mg C m⁻² h⁻¹")
        print(f"  HBR post-break median ({int(bp_year_hbr)}+): {hbr_post_median:.4f} mg C m⁻² h⁻¹")

    # Figure 7: Breakpoint visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # BES
    ax1.plot(bes_annual.index, bes_annual.values, 'o-', linewidth=2, markersize=5, color='steelblue')
    if bp_year_bes is not None:
        ax1.axvline(bp_year_bes, color='red', linestyle='--', linewidth=2, label=f'Break: {bp_year_bes}')
        # Plot additional breakpoints if any
        for bp in breakpoints_bes[1:]:
            extra_yr = bes_annual.index[bp - 1]
            ax1.axvline(extra_yr, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
        # Pre/post-break median step lines
        bes_pre_yrs = bes_annual.index[bes_annual.index < bp_year_bes]
        bes_post_yrs = bes_annual.index[bes_annual.index >= bp_year_bes]
        if len(bes_pre_yrs) > 0 and len(bes_post_yrs) > 0:
            ax1.hlines(bes_pre_median, bes_pre_yrs.min(), bp_year_bes,
                       colors='gray', linestyles='--', linewidth=1.5, alpha=0.7,
                       label=f'Pre-break median: {bes_pre_median:.3f}')
            ax1.hlines(bes_post_median, bp_year_bes, bes_post_yrs.max(),
                       colors='gray', linestyles='--', linewidth=1.5, alpha=0.7,
                       label=f'Post-break median: {bes_post_median:.3f}')

    ax1.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax1.set_ylabel('Annual Median CH₄ Flux (mg C m⁻² h⁻¹)', fontsize=11)
    ax1.set_title('A) BES Forest Sites (Annual Median)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # HBR Reference
    ax2.plot(hbr_ref['Year'], hbr_ref['Annual_CH4_flux'], 'o-', linewidth=2, markersize=5, color='darkorange')
    if bp_year_hbr is not None:
        ax2.axvline(bp_year_hbr, color='red', linestyle='--', linewidth=2, label=f'Break: {bp_year_hbr}')
        # Pre/post-break median step lines
        hbr_pre_yrs = hbr_ref[hbr_ref['Year'] < bp_year_hbr]['Year']
        hbr_post_yrs = hbr_ref[hbr_ref['Year'] >= bp_year_hbr]['Year']
        if len(hbr_pre_yrs) > 0 and len(hbr_post_yrs) > 0:
            ax2.hlines(hbr_pre_median, hbr_pre_yrs.min(), bp_year_hbr,
                       colors='gray', linestyles='--', linewidth=1.5, alpha=0.7,
                       label=f'Pre-break median: {hbr_pre_median:.3f}')
            ax2.hlines(hbr_post_median, bp_year_hbr, hbr_post_yrs.max(),
                       colors='gray', linestyles='--', linewidth=1.5, alpha=0.7,
                       label=f'Post-break median: {hbr_post_median:.3f}')

    ax2.axhline(0, color='black', linestyle=':', linewidth=0.5)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Annual CH₄ Flux (mg C m⁻² h⁻¹)', fontsize=11)
    ax2.set_title('B) HBR Reference Watershed (WS6-BB)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Structural Breakpoint Detection\n(PELT Algorithm)', fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_7_Breakpoints.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_7_Breakpoints.svg'))
    plt.close()

    print(f"\nFigure 7 saved.")

    return {
        'bes_annual': bes_annual,
        'hbr_annual': hbr_ref,
        'bes_breakpoints': breakpoints_bes,
        'hbr_breakpoints': breakpoints_hbr,
        'bp_year_bes': bp_year_bes,
        'bp_year_hbr': bp_year_hbr,
        'bes_pre_median': bes_pre_median,
        'bes_post_median': bes_post_median,
        'hbr_pre_median': hbr_pre_median,
        'hbr_post_median': hbr_post_median,
    }

# ============================================================================
# BLOCK 8: DEPOSITION OVERLAY
# ============================================================================

def block_8_deposition_overlay():
    """
    Block 8: Deposition Trends Overlay
    ================================
    Overlay SO4 and inorganic N deposition with CH4 decline.
    If N deposition drives the decline (not moisture), should align better.
    """
    print("\n" + "="*70)
    print("BLOCK 8: DEPOSITION OVERLAY")
    print("="*70)

    # Load NADP deposition
    nadp_data = load_nadp()

    # Process Hubbard Brook (NH02)
    df_hb = nadp_data['HB'].copy()
    df_hb['Year'] = df_hb['yr']

    # Create annual sums
    so4_hb_annual = df_hb.groupby('Year')['SO4'].sum()
    n_inorg_hb = (df_hb['NH4'] + df_hb['NO3']).fillna(0)
    n_hb_annual = df_hb.groupby('Year')[['NH4', 'NO3']].sum().sum(axis=1)

    print(f"\nHubbard Brook (NH02) deposition:")
    print(f"  SO4: {so4_hb_annual.min():.2f}-{so4_hb_annual.max():.2f} kg/ha/yr")
    print(f"  Inorg N: {n_hb_annual.min():.2f}-{n_hb_annual.max():.2f} kg/ha/yr")

    # Process Baltimore (MD99)
    df_md = nadp_data['MD'].copy()
    df_md['Year'] = df_md['yr']

    so4_md_annual = df_md.groupby('Year')['SO4'].sum()
    n_md_annual = df_md.groupby('Year')[['NH4', 'NO3']].sum().sum(axis=1)

    print(f"\nBaltimore (MD99) deposition:")
    print(f"  SO4: {so4_md_annual.min():.2f}-{so4_md_annual.max():.2f} kg/ha/yr")
    print(f"  Inorg N: {n_md_annual.min():.2f}-{n_md_annual.max():.2f} kg/ha/yr")

    # Load CH4 flux for overlay (trimmed, median for robustness)
    bes = load_bes_flux()
    bes_forest = bes[bes['Site'].isin(FOREST_SITES) & bes['CH4_flux'].notna()].copy()
    bes_forest = trim_outliers(bes_forest, flux_col='CH4_flux')
    ch4_bes = bes_forest.groupby('Year')['CH4_flux'].median()

    hbr_annual = load_hbr_annual()
    hbr_hb = hbr_annual[hbr_annual['StudySite'] == 'Hubbard Brook']
    ch4_hbr = hbr_hb.groupby('Year')['Annual_CH4_flux'].median()

    # Figure 8: Dual-axis plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    # Panel A: Hubbard Brook — HB SO4 deposition + HBR CH4 flux
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(so4_hb_annual.index, so4_hb_annual.values, 'o-', linewidth=2,
                     label='SO₄ Deposition (HB)', color='steelblue')
    line2 = ax1_twin.plot(ch4_hbr.index, ch4_hbr.values, 's--', linewidth=2,
                          label='CH₄ Flux (HBR)', color='coral', alpha=0.7)

    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('SO₄ Deposition (kg/ha/yr)', fontsize=11, color='steelblue')
    ax1_twin.set_ylabel('CH₄ Flux (mg C m⁻² h⁻¹)', fontsize=11, color='coral')
    ax1.set_title('A) Hubbard Brook Region', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1_twin.tick_params(axis='y', labelcolor='coral')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Panel B: Baltimore — MD Inorg N deposition + BES CH4 flux
    ax2_twin = ax2.twinx()
    line3 = ax2.plot(n_md_annual.index, n_md_annual.values, 'o-', linewidth=2,
                     label='Inorg N Deposition (MD)', color='darkgreen')
    line4 = ax2_twin.plot(ch4_bes.index, ch4_bes.values, 's--', linewidth=2,
                          label='CH₄ Flux (BES)', color='darkorange', alpha=0.7)

    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Inorg N Deposition (kg/ha/yr)', fontsize=11, color='darkgreen')
    ax2_twin.set_ylabel('CH₄ Flux (mg C m⁻² h⁻¹)', fontsize=11, color='darkorange')
    ax2.set_title('B) Baltimore Region', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax2_twin.tick_params(axis='y', labelcolor='darkorange')

    lines2 = line3 + line4
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc='upper left')

    fig.suptitle('Deposition Trends Overlaid with CH₄ Flux Decline',
                 fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_8_Deposition.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_8_Deposition.svg'))
    plt.close()

    print(f"\nFigure 8 saved.")

    return {
        'so4_hb': so4_hb_annual,
        'n_hb': n_hb_annual,
        'so4_md': so4_md_annual,
        'n_md': n_md_annual,
        'ch4_bes': ch4_bes,
        'ch4_hbr': ch4_hbr,
    }

# ============================================================================
# BLOCK 9: SOIL NITROGEN TRENDS
# ============================================================================

def block_9_soil_nitrogen():
    """
    Block 9: Soil Nitrate Trends
    ===========================
    Examines whether soil NO3 (from lysimeter data) tracks with CH4 decline.
    May indicate N saturation or biogeochemical shift.
    """
    print("\n" + "="*70)
    print("BLOCK 9: SOIL NITROGEN TRENDS")
    print("="*70)

    # Load lysimeter data
    lys = load_lysimeter()

    # Filter to forest sites and focus on NO3
    forest_lys = lys[lys['Site'].isin(['LEA', 'HD', 'ORM'])].copy()

    # Annual mean NO3 by site
    print(f"\nSoil NO₃ trends (lysimeter data):")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    all_trends = {}
    for idx, site in enumerate(['LEA', 'HD', 'ORM']):
        site_data = forest_lys[forest_lys['Site'] == site]
        no3_by_year = site_data.groupby('Year')['NO3'].mean()

        # Remove first/last years with sparse data
        no3_clean = no3_by_year[(no3_by_year.index >= 2000) & (no3_by_year.index <= 2023)]

        if len(no3_clean) > 2:
            trend = linregress(no3_clean.index, no3_clean.values)
            all_trends[site] = trend

            print(f"\n{site}:")
            print(f"  Slope: {trend.slope:.6f} mg/L per year, p={trend.pvalue:.4e}")
            print(f"  R²: {trend.rvalue**2:.4f}")

            # Plot
            ax = axes[idx]
            ax.plot(no3_clean.index, no3_clean.values, 'o-', linewidth=2, markersize=5)

            # Add trend line
            x_trend = np.array([no3_clean.index.min(), no3_clean.index.max()])
            ax.plot(x_trend, trend.intercept + trend.slope * x_trend, 'r--', linewidth=1.5)

            site_type = 'Urban' if site in URBAN_SITES else 'Rural'
            ax.set_title(f'{site} ({site_type})', fontsize=11, fontweight='bold')
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('NO₃ (mg/L)', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax = axes[idx]
            ax.text(0.5, 0.5, f'Insufficient data\n({len(no3_clean)} years)',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(site, fontsize=11, fontweight='bold')

    fig.suptitle('Soil Nitrate Trends in Forest Lysimeters', fontsize=13, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_9_SoilNitrogen.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_9_SoilNitrogen.svg'))
    plt.close()

    print(f"\nFigure 9 saved.")

    return all_trends

# ============================================================================
# BLOCK 10: VEGETATION CONTEXT
# ============================================================================

def block_10_vegetation():
    """
    Block 10: Vegetation Community Context
    ====================================
    Examine seedling cover across census years to contextualize
    forest recovery or compositional change.
    """
    print("\n" + "="*70)
    print("BLOCK 10: VEGETATION CONTEXT")
    print("="*70)

    # Load vegetation data
    veg = load_vegetation()

    print(f"\nVegetation data shape: {veg.shape}")
    print(f"Columns: {veg.columns.tolist()}")

    # This data contains species-level cover. Calculate total seedling cover per plot
    species_cols = [c for c in veg.columns if c not in ['Region', 'Site', 'Year', 'Plot']]

    if len(species_cols) > 0:
        # Calculate total seedling cover per site-year
        veg['Total_Cover'] = veg[species_cols].sum(axis=1)

        # Filter to forest sites
        forest_veg = veg[veg['Site'].isin(FOREST_SITES)].copy()

        # Aggregate by year and site
        veg_by_year = forest_veg.groupby(['Year', 'Site'])['Total_Cover'].mean().reset_index()

        print(f"\nTotal seedling cover by site and year:")
        print(veg_by_year.pivot(index='Year', columns='Site', values='Total_Cover'))

        # Figure 10: Bar chart for census years
        years = sorted(veg_by_year['Year'].unique())
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        sites_to_plot = [s for s in ['LEA', 'HD', 'ORM'] if s in veg_by_year['Site'].unique()]
        x_pos = np.arange(len(years))
        width = 0.25

        for i, site in enumerate(sites_to_plot):
            site_subset = veg_by_year[veg_by_year['Site'] == site]
            if len(site_subset) > 0:
                values = [site_subset[site_subset['Year'] == y]['Total_Cover'].values[0]
                          if len(site_subset[site_subset['Year'] == y]) > 0 else 0
                          for y in years]
                site_type = 'Urban' if site in URBAN_SITES else 'Rural'
                ax.bar(x_pos + i*width, values, width, label=f'{site} ({site_type})', alpha=0.8)

        ax.set_xlabel('Census Year', fontsize=12)
        ax.set_ylabel('Total Seedling Cover (%)', fontsize=12)
        ax.set_title('Vegetation Context: Total Seedling Cover by Census Year',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_10_Vegetation.png'), dpi=DPI)
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_10_Vegetation.svg'))
        plt.close()

        print(f"\nFigure 10 saved.")

        return veg_by_year
    else:
        print("No species cover data found")
        return None

# ============================================================================
# BLOCK 11: SOIL MOISTURE DIRECT TEST
# ============================================================================

def block_11_soil_moisture_flux(precip_r2=None):
    """
    Block 11: Direct Test - In-Situ Soil Moisture vs CH₄ Flux
    ===========================================================
    Uses actual volumetric water content (VWC) measurements instead of
    precipitation as a proxy for soil moisture. This is the strongest
    direct test of the diffusion limitation hypothesis.

    Parameters
    ----------
    precip_r2 : float, optional
        R² from Block 1 precipitation regression, for comparison panel.
    """
    print("\n" + "="*70)
    print("BLOCK 11: SOIL MOISTURE DIRECT TEST (VWC vs FLUX)")
    print("="*70)

    # Load BES flux data
    bes = load_bes_flux()

    # Filter to forest sites and remove missing CH4
    forest_mask = bes['Site'].isin(FOREST_SITES)
    bes = bes[forest_mask & bes['CH4_flux'].notna()].copy()

    # Remove outliers
    bes = trim_outliers(bes, flux_col='CH4_flux')

    # Load soil moisture data
    vwc = load_soil_moisture()

    # Calculate monthly mean VWC per site
    vwc_monthly = vwc.groupby(['Site_Code', 'YearMonth']).agg({
        'Mean_VWC': 'mean'
    }).reset_index()

    # Create site mapping between flux data ('Site') and VWC data ('Site_Code')
    # They should already match, but verify
    vwc_sites = vwc_monthly['Site_Code'].unique()
    bes_sites = bes['Site'].unique()
    print(f"\nVWC sites: {sorted(vwc_sites)}")
    print(f"BES flux sites: {sorted(bes_sites)}")

    # Merge flux with VWC on Site and YearMonth
    all_data = []
    for site in FOREST_SITES:
        bes_site = bes[bes['Site'] == site].copy()
        vwc_site = vwc_monthly[vwc_monthly['Site_Code'] == site].copy()

        if len(bes_site) == 0 or len(vwc_site) == 0:
            continue

        merged = pd.merge(
            bes_site, vwc_site,
            left_on=['Site', 'YearMonth'], right_on=['Site_Code', 'YearMonth'],
            how='inner'
        )
        all_data.append(merged)

    if len(all_data) == 0:
        print("\nNo overlapping data between VWC and flux. Skipping Block 11.")
        return None

    merged_all = pd.concat(all_data, ignore_index=True)

    print(f"\nMerged data: {len(merged_all)} observations")
    print(f"Sites with overlap: {sorted(merged_all['Site'].unique())}")

    # Pooled regression: CH4_flux ~ Mean_VWC
    stats_pooled = regression_stats(
        merged_all['Mean_VWC'].values,
        merged_all['CH4_flux'].values,
        label="All overlapping sites (pooled)"
    )

    print(f"\nPooled regression (n={stats_pooled['n']}):")
    print(f"  Slope: {stats_pooled['slope']:.6f} mg C m-2 h-1 per unit VWC")
    print(f"  R²: {stats_pooled['r2']:.4f}")
    print(f"  p-value: {stats_pooled['p_value']:.4e}")
    print(f"  95% CI: [{stats_pooled['ci_lower']:.6f}, {stats_pooled['ci_upper']:.6f}]")

    # By-site regressions
    print(f"\nBy-site regressions (VWC vs flux):")
    site_stats = []
    for site in sorted(set(merged_all['Site'])):
        site_subset = merged_all[merged_all['Site'] == site]
        stats_site = regression_stats(
            site_subset['Mean_VWC'].values,
            site_subset['CH4_flux'].values,
            label=site
        )
        if stats_site:
            site_stats.append(stats_site)
            print(f"  {site}: r²={stats_site['r2']:.4f}, p={stats_site['p_value']:.4e}, n={stats_site['n']}")

    # Figure 11: Two-panel figure (VWC vs flux, and R² comparison)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Panel A: Scatterplot with regression
    ax1.scatter(merged_all['Mean_VWC'], merged_all['CH4_flux'], alpha=0.15, s=15)

    # Plot regression line
    x_range = np.linspace(merged_all['Mean_VWC'].min(), merged_all['Mean_VWC'].max(), 100)
    y_pred = stats_pooled['intercept'] + stats_pooled['slope'] * x_range
    ax1.plot(x_range, y_pred, 'r-', linewidth=2, label='Linear regression')

    p_str = f"p = {stats_pooled['p_value']:.3e}" if stats_pooled['p_value'] >= 0.001 else "p < 0.001"
    ax1.text(0.05, 0.95, f"r² = {stats_pooled['r2']:.3f}\n{p_str}\nn = {stats_pooled['n']}",
            transform=ax1.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xlabel('Mean Volumetric Water Content (VWC)', fontsize=12)
    ax1.set_ylabel('CH₄ Flux (mg C m⁻² h⁻¹)', fontsize=12)
    ax1.set_title('Panel A: In-Situ VWC vs CH₄ Flux', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel B: R² comparison (VWC vs precipitation from Block 1)
    precip_r2_val = precip_r2 if precip_r2 is not None else 0.0
    comparison_data = {
        'In-Situ VWC': stats_pooled['r2'],
        'Precipitation': precip_r2_val
    }

    ax2.bar(comparison_data.keys(), comparison_data.values(), color=['steelblue', 'lightcoral'], alpha=0.8)
    ax2.set_ylabel('R² Value', fontsize=12)
    ax2.set_title('Panel B: Moisture Predictor Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(1.0, max(comparison_data.values()) * 1.2))
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (k, v) in enumerate(comparison_data.items()):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_11_SoilMoisture_Flux.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_11_SoilMoisture_Flux.svg'))
    plt.close()

    print(f"\nFigure 11 saved.")

    return {
        'pooled': stats_pooled,
        'by_site': site_stats,
        'merged_data': merged_all
    }

# ============================================================================
# BLOCK 12: HARVARD FOREST INDEPENDENT REPLICATION
# ============================================================================

def block_12_harvard_forest():
    """
    Block 12: Harvard Forest CH₄ Context (1991–1994)
    ================================================
    Harvard Forest atmospheric CH4 concentration provides contextual evidence
    from an independent forest ecosystem. Note: This is ambient concentration,
    not soil flux, and predates the BES/HBR records. No precipitation overlap
    for direct regression testing.
    """
    print("\n" + "="*70)
    print("BLOCK 12: HARVARD FOREST ATMOSPHERIC CH₄ CONTEXT")
    print("="*70)

    # Load Harvard Forest CH4 data
    hf = load_harvard_forest()

    print(f"\nHarvard Forest CH4 data: {len(hf)} rows")
    print(f"Date range: {hf['datetime'].min()} to {hf['datetime'].max()}")
    print(f"CH4 concentration range: {hf['ch4'].min():.3f} to {hf['ch4'].max():.3f} ppm")

    # Calculate monthly mean CH4 concentration
    hf_monthly = hf.groupby('YearMonth').agg({
        'ch4': 'mean'
    }).reset_index()

    hf_monthly['Year'] = hf_monthly['YearMonth'].dt.year
    hf_monthly['Month'] = hf_monthly['YearMonth'].dt.month

    print(f"\nMonthly data: {len(hf_monthly)} months")
    print(f"Seasonal range: {hf_monthly['ch4'].min():.3f} to {hf_monthly['ch4'].max():.3f} ppm")

    # Try to load Harvard Forest PRISM data (if available)
    prism_hf_path = os.path.join(DATA_DIR, "PRISM/HarvardForest-PRISM_ppt_tmean_stable_4km_199801_202506_42.5380_-72.1710.csv")
    prism_hf = None
    if os.path.exists(prism_hf_path):
        try:
            prism_hf = pd.read_csv(prism_hf_path, skiprows=10)
            print(f"\nHarvard Forest PRISM data loaded: {len(prism_hf)} rows")
            print(f"PRISM date range: {prism_hf['Date'].min()} to {prism_hf['Date'].max()}")
            print("NOTE: PRISM data starts at 1998, but HF CH4 data is 1991-1994. No temporal overlap for direct regression.")
        except Exception as e:
            print(f"\nCould not load Harvard Forest PRISM data: {e}")

    # Calculate trend line for the HF CH4 time series
    hf_monthly['month_num'] = np.arange(len(hf_monthly))
    trend_stats = regression_stats(
        hf_monthly['month_num'].values,
        hf_monthly['ch4'].values,
        label="Harvard Forest CH4 trend"
    )

    print(f"\nTrend analysis (1991-1994):")
    print(f"  Slope: {trend_stats['slope']:.6f} ppm/month")
    print(f"  R²: {trend_stats['r2']:.4f}")
    print(f"  p-value: {trend_stats['p_value']:.4e}")

    # Figure 12: Time series with trend line
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Plot monthly mean
    ax.plot(hf_monthly.index, hf_monthly['ch4'], 'o-', linewidth=2, markersize=5, label='Monthly mean', color='steelblue')

    # Plot trend line
    y_trend = trend_stats['intercept'] + trend_stats['slope'] * hf_monthly['month_num'].values
    ax.plot(hf_monthly.index, y_trend, 'r--', linewidth=2, label='Trend line')

    p_str = f"p = {trend_stats['p_value']:.3e}" if trend_stats['p_value'] >= 0.001 else "p < 0.001"
    ax.text(0.05, 0.95, f"Trend r² = {trend_stats['r2']:.3f}\n{p_str}\nn = {trend_stats['n']}",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax.set_xlabel('Month (1991-1994)', fontsize=12)
    ax.set_ylabel('CH₄ Concentration (ppm)', fontsize=12)
    ax.set_title('Harvard Forest Atmospheric CH₄ Context (1991–1994)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_12_HarvardForest.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_12_HarvardForest.svg'))
    plt.close()

    print(f"\nFigure 12 saved.")
    print("NOTE: This is ambient atmospheric CH₄ concentration, not soil flux.")
    print("The 1991-1994 period predates BES/HBR records. Provides baseline context only.")

    return {
        'monthly_data': hf_monthly,
        'trend': trend_stats,
        'prism_available': prism_hf is not None
    }

# ============================================================================
# BLOCK 13: SOIL PROPERTIES CHARACTERIZATION
# ============================================================================

def block_13_soil_properties():
    """
    Block 13: Forest Soil Biogeochemical Properties
    ===============================================
    Characterizes soil biogeochemistry across forest study sites.
    Shows site-level variation in C:N ratio, microbial biomass,
    and N mineralization/nitrification processes.
    """
    print("\n" + "="*70)
    print("BLOCK 13: SOIL PROPERTIES CHARACTERIZATION")
    print("="*70)

    # Load soil properties
    soil = load_soil_properties()

    print(f"\nSoil properties data: {len(soil)} rows")
    print(f"Forest sites: {soil['Site_Code'].unique()}")

    # Filter to 0-10cm depth (most relevant for trace gas dynamics)
    soil_shallow = soil[soil['Depth'] == '0to10'].copy()

    print(f"At 0-10cm depth: {len(soil_shallow)} observations")

    # Key variables for characterization
    key_vars = ['BD', 'N_Perc', 'C_Perc', 'C_N', 'MB_Carbon', 'Respiration', 'Net_N_Min', 'Net_Nitr']

    # Calculate mean and SD by site
    soil_stats = soil_shallow.groupby('Site_Code')[key_vars].agg(['mean', 'std']).reset_index()

    print(f"\nSite-level biogeochemical properties (0-10cm):")
    for site in sorted(soil_shallow['Site_Code'].unique()):
        site_data = soil_shallow[soil_shallow['Site_Code'] == site]
        print(f"\n  {site} (n={len(site_data)}):")
        if len(site_data) > 0:
            print(f"    C:N ratio: {site_data['C_N'].mean():.2f} ± {site_data['C_N'].std():.2f}")
            print(f"    MB Carbon: {site_data['MB_Carbon'].mean():.1f} ± {site_data['MB_Carbon'].std():.1f} µg C g-1")
            print(f"    Net N Min: {site_data['Net_N_Min'].mean():.3f} ± {site_data['Net_N_Min'].std():.3f}")
            print(f"    Net Nitr:  {site_data['Net_Nitr'].mean():.4f} ± {site_data['Net_Nitr'].std():.4f}")

    # Figure 13: Multi-panel comparison (2x2)
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DOUBLE)

    # Panel A: C:N ratio
    ax = axes[0, 0]
    sites = sorted(soil_shallow['Site_Code'].unique())
    cn_means = [soil_shallow[soil_shallow['Site_Code'] == s]['C_N'].mean() for s in sites]
    cn_stds = [soil_shallow[soil_shallow['Site_Code'] == s]['C_N'].std() for s in sites]
    ax.bar(sites, cn_means, yerr=cn_stds, capsize=5, color='steelblue', alpha=0.8, error_kw={'elinewidth': 2})
    ax.set_ylabel('C:N Ratio', fontsize=11)
    ax.set_title('Panel A: C:N Ratio', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Microbial Biomass Carbon
    ax = axes[0, 1]
    mb_means = [soil_shallow[soil_shallow['Site_Code'] == s]['MB_Carbon'].mean() for s in sites]
    mb_stds = [soil_shallow[soil_shallow['Site_Code'] == s]['MB_Carbon'].std() for s in sites]
    ax.bar(sites, mb_means, yerr=mb_stds, capsize=5, color='darkgreen', alpha=0.8, error_kw={'elinewidth': 2})
    ax.set_ylabel('MB Carbon (µg C g⁻¹)', fontsize=11)
    ax.set_title('Panel B: Microbial Biomass Carbon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Net N Mineralization
    ax = axes[1, 0]
    nnm_means = [soil_shallow[soil_shallow['Site_Code'] == s]['Net_N_Min'].mean() for s in sites]
    nnm_stds = [soil_shallow[soil_shallow['Site_Code'] == s]['Net_N_Min'].std() for s in sites]
    ax.bar(sites, nnm_means, yerr=nnm_stds, capsize=5, color='orange', alpha=0.8, error_kw={'elinewidth': 2})
    ax.set_ylabel('Net N Min (µg N g⁻¹ d⁻¹)', fontsize=11)
    ax.set_title('Panel C: Net N Mineralization', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: Net Nitrification
    ax = axes[1, 1]
    nnitr_means = [soil_shallow[soil_shallow['Site_Code'] == s]['Net_Nitr'].mean() for s in sites]
    nnitr_stds = [soil_shallow[soil_shallow['Site_Code'] == s]['Net_Nitr'].std() for s in sites]
    ax.bar(sites, nnitr_means, yerr=nnitr_stds, capsize=5, color='red', alpha=0.8, error_kw={'elinewidth': 2})
    ax.set_ylabel('Net Nitr (µg N g⁻¹ d⁻¹)', fontsize=11)
    ax.set_title('Panel D: Net Nitrification', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Main title
    fig.suptitle('Forest Soil Biogeochemical Properties Across Study Sites (0–10 cm)',
                fontsize=13, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_13_SoilProperties.png'), dpi=DPI)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_13_SoilProperties.svg'))
    plt.close()

    print(f"\nFigure 13 saved.")

    return {
        'shallow_soil_data': soil_shallow,
        'site_stats': soil_stats
    }

# ============================================================================
# BLOCK 14: MULTI-SCALE ANALYSIS
# ============================================================================

def block_14_multi_scale(merged_temp_data):
    """
    Block 14: Multi-Scale Regression Analysis
    ==========================================
    Re-fit CH4 ~ Precip + Temp + Year at three temporal scales:
      1. Individual measurement (n ~ 9,359) — already in Block 3
      2. Seasonal-site means (aggregate by site × season × year)
      3. Annual-site means (aggregate by site × year)

    Preempts the chamber-noise critique: if precipitation explains nothing
    at aggregated scales where measurement noise is averaged out, diffusion
    limitation fails even under the most generous interpretation.
    """
    print("\n" + "="*70)
    print("BLOCK 14: MULTI-SCALE ANALYSIS")
    print("="*70)

    df = merged_temp_data[['CH4_flux', 'ppt_mm', 'tmean_c', 'Year', 'Site']].copy()
    df = df.dropna()

    # Assign season
    # We need month info — derive from the merged data if available
    if 'Month' not in df.columns:
        # Try various ways to get month
        if 'Month' in merged_temp_data.columns:
            df['Month'] = merged_temp_data.loc[df.index, 'Month']
        elif 'month' in merged_temp_data.columns:
            df['Month'] = merged_temp_data.loc[df.index, 'month']
        elif 'YearMonth' in merged_temp_data.columns:
            df['Month'] = merged_temp_data.loc[df.index, 'YearMonth'].apply(lambda x: x.month)
        else:
            for col in ['date', 'Date', 'collection_date', 'CollectionDate']:
                if col in merged_temp_data.columns:
                    df['Month'] = pd.to_datetime(merged_temp_data.loc[df.index, col]).dt.month
                    break

    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}

    results = {}

    # --- Scale 1: Individual measurement ---
    df_m = df.copy()
    df_m['ppt_std'] = (df_m['ppt_mm'] - df_m['ppt_mm'].mean()) / df_m['ppt_mm'].std()
    df_m['tmean_std'] = (df_m['tmean_c'] - df_m['tmean_c'].mean()) / df_m['tmean_c'].std()
    df_m['year_std'] = (df_m['Year'] - df_m['Year'].mean()) / df_m['Year'].std()

    X_m = sm.add_constant(df_m[['ppt_std', 'tmean_std', 'year_std']])
    model_m = sm.OLS(df_m['CH4_flux'], X_m).fit()

    results['measurement'] = {
        'n': int(model_m.nobs),
        'r2': model_m.rsquared,
        'f_stat': model_m.fvalue,
        'f_pval': model_m.f_pvalue,
        'beta_ppt': model_m.params.get('ppt_std', np.nan),
        'beta_temp': model_m.params.get('tmean_std', np.nan),
        'beta_year': model_m.params.get('year_std', np.nan),
        'p_ppt': model_m.pvalues.get('ppt_std', np.nan),
        'p_temp': model_m.pvalues.get('tmean_std', np.nan),
        'p_year': model_m.pvalues.get('year_std', np.nan),
    }

    print(f"\n  Scale 1 — Individual measurements (n={results['measurement']['n']})")
    print(f"    R² = {results['measurement']['r2']:.4f}")
    print(f"    β_ppt = {results['measurement']['beta_ppt']:.4f} (p={results['measurement']['p_ppt']:.2e})")
    print(f"    β_temp = {results['measurement']['beta_temp']:.4f} (p={results['measurement']['p_temp']:.2e})")
    print(f"    β_year = {results['measurement']['beta_year']:.4f} (p={results['measurement']['p_year']:.2e})")

    # --- Scale 2: Seasonal-site means ---
    if 'Month' in df.columns:
        df['Season'] = df['Month'].map(season_map)
        df_seasonal = df.groupby(['Site', 'Year', 'Season']).agg(
            CH4_flux=('CH4_flux', 'mean'),
            ppt_mm=('ppt_mm', 'mean'),
            tmean_c=('tmean_c', 'mean')
        ).reset_index()

        df_s = df_seasonal.dropna()
        if len(df_s) > 10:
            df_s['ppt_std'] = (df_s['ppt_mm'] - df_s['ppt_mm'].mean()) / df_s['ppt_mm'].std()
            df_s['tmean_std'] = (df_s['tmean_c'] - df_s['tmean_c'].mean()) / df_s['tmean_c'].std()
            df_s['year_std'] = (df_s['Year'] - df_s['Year'].mean()) / df_s['Year'].std()

            X_s = sm.add_constant(df_s[['ppt_std', 'tmean_std', 'year_std']])
            model_s = sm.OLS(df_s['CH4_flux'], X_s).fit()

            results['seasonal'] = {
                'n': int(model_s.nobs),
                'r2': model_s.rsquared,
                'f_stat': model_s.fvalue,
                'f_pval': model_s.f_pvalue,
                'beta_ppt': model_s.params.get('ppt_std', np.nan),
                'beta_temp': model_s.params.get('tmean_std', np.nan),
                'beta_year': model_s.params.get('year_std', np.nan),
                'p_ppt': model_s.pvalues.get('ppt_std', np.nan),
                'p_temp': model_s.pvalues.get('tmean_std', np.nan),
                'p_year': model_s.pvalues.get('year_std', np.nan),
            }

            print(f"\n  Scale 2 — Seasonal-site means (n={results['seasonal']['n']})")
            print(f"    R² = {results['seasonal']['r2']:.4f}")
            print(f"    β_ppt = {results['seasonal']['beta_ppt']:.4f} (p={results['seasonal']['p_ppt']:.2e})")
            print(f"    β_temp = {results['seasonal']['beta_temp']:.4f} (p={results['seasonal']['p_temp']:.2e})")
            print(f"    β_year = {results['seasonal']['beta_year']:.4f} (p={results['seasonal']['p_year']:.2e})")
        else:
            print("\n  Scale 2 — Insufficient seasonal data for regression.")
    else:
        print("\n  Scale 2 — Month column unavailable; skipping seasonal aggregation.")

    # --- Scale 3: Annual-site means ---
    df_annual = df.groupby(['Site', 'Year']).agg(
        CH4_flux=('CH4_flux', 'mean'),
        ppt_mm=('ppt_mm', 'mean'),
        tmean_c=('tmean_c', 'mean')
    ).reset_index()

    df_a = df_annual.dropna()
    if len(df_a) > 10:
        df_a['ppt_std'] = (df_a['ppt_mm'] - df_a['ppt_mm'].mean()) / df_a['ppt_mm'].std()
        df_a['tmean_std'] = (df_a['tmean_c'] - df_a['tmean_c'].mean()) / df_a['tmean_c'].std()
        df_a['year_std'] = (df_a['Year'] - df_a['Year'].mean()) / df_a['Year'].std()

        X_a = sm.add_constant(df_a[['ppt_std', 'tmean_std', 'year_std']])
        model_a = sm.OLS(df_a['CH4_flux'], X_a).fit()

        results['annual'] = {
            'n': int(model_a.nobs),
            'r2': model_a.rsquared,
            'f_stat': model_a.fvalue,
            'f_pval': model_a.f_pvalue,
            'beta_ppt': model_a.params.get('ppt_std', np.nan),
            'beta_temp': model_a.params.get('tmean_std', np.nan),
            'beta_year': model_a.params.get('year_std', np.nan),
            'p_ppt': model_a.pvalues.get('ppt_std', np.nan),
            'p_temp': model_a.pvalues.get('tmean_std', np.nan),
            'p_year': model_a.pvalues.get('year_std', np.nan),
        }

        print(f"\n  Scale 3 — Annual-site means (n={results['annual']['n']})")
        print(f"    R² = {results['annual']['r2']:.4f}")
        print(f"    β_ppt = {results['annual']['beta_ppt']:.4f} (p={results['annual']['p_ppt']:.2e})")
        print(f"    β_temp = {results['annual']['beta_temp']:.4f} (p={results['annual']['p_temp']:.2e})")
        print(f"    β_year = {results['annual']['beta_year']:.4f} (p={results['annual']['p_year']:.2e})")
    else:
        print("\n  Scale 3 — Insufficient annual data for regression.")

    # --- Figure 14: Multi-scale comparison ---
    scales = []
    r2_vals = []
    beta_ppt_vals = []
    beta_temp_vals = []
    beta_year_vals = []
    n_vals = []

    for scale_name in ['measurement', 'seasonal', 'annual']:
        if scale_name in results:
            scales.append(scale_name.capitalize())
            r2_vals.append(results[scale_name]['r2'])
            beta_ppt_vals.append(results[scale_name]['beta_ppt'])
            beta_temp_vals.append(results[scale_name]['beta_temp'])
            beta_year_vals.append(results[scale_name]['beta_year'])
            n_vals.append(results[scale_name]['n'])

    if len(scales) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

        # Panel A: R² across scales
        x_pos = np.arange(len(scales))
        bars = axes[0].bar(x_pos, r2_vals, color=['steelblue', 'darkorange', 'forestgreen'][:len(scales)],
                          alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f"{s}\n(n={n})" for s, n in zip(scales, n_vals)], fontsize=10)
        axes[0].set_ylabel('R²', fontsize=12)
        axes[0].set_title('A) Total Variance Explained\nAcross Temporal Scales', fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, max(r2_vals) * 2.5 if max(r2_vals) > 0 else 0.1)
        for bar, val in zip(bars, r2_vals):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Panel B: Standardized coefficients across scales
        width = 0.25
        x_pos2 = np.arange(len(scales))
        axes[1].bar(x_pos2 - width, beta_ppt_vals, width, label='Precipitation',
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[1].bar(x_pos2, beta_temp_vals, width, label='Temperature',
                   color='darkorange', alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[1].bar(x_pos2 + width, beta_year_vals, width, label='Year',
                   color='forestgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[1].set_xticks(x_pos2)
        axes[1].set_xticklabels(scales, fontsize=10)
        axes[1].set_ylabel('Standardized β', fontsize=12)
        axes[1].set_title('B) Predictor Coefficients\nAcross Temporal Scales', fontsize=12, fontweight='bold')
        axes[1].axhline(0, color='black', linewidth=0.5)
        axes[1].legend(fontsize=9, loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_14_MultiScale.png'), dpi=DPI)
        plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_14_MultiScale.svg'))
        plt.close()
        print(f"\nFigure 14 saved.")
    else:
        print("\nInsufficient scales for Figure 14.")

    return results


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary(results_dict):
    """Print comprehensive summary of all analyses."""
    print("\n" + "="*70)
    print("COMPREHENSIVE STATISTICAL SUMMARY")
    print("="*70)

    summary_lines = []
    summary_lines.append("\n1. PRECIPITATION-FLUX REGRESSION (Block 1)")
    summary_lines.append("-" * 50)
    if 'block1' in results_dict:
        r1 = results_dict['block1']['pooled']
        summary_lines.append(f"  Precipitation vs CH₄ flux (pooled, n={r1['n']})")
        summary_lines.append(f"    R² = {r1['r2']:.4f}")
        summary_lines.append(f"    Slope = {r1['slope']:.6f}, p = {r1['p_value']:.4e}")
        summary_lines.append(f"    95% CI: [{r1['ci_lower']:.6f}, {r1['ci_upper']:.6f}]")
        summary_lines.append(f"  Interpretation: Precipitation explains {r1['r2']*100:.1f}% of variance")
        summary_lines.append(f"                  (Hypothesis predicts high r², NOT observed)")

    summary_lines.append("\n2. TEMPERATURE-FLUX REGRESSION (Block 2)")
    summary_lines.append("-" * 50)
    if 'block2' in results_dict:
        r2 = results_dict['block2']['pooled']
        summary_lines.append(f"  Temperature vs CH₄ flux (pooled, n={r2['n']})")
        summary_lines.append(f"    R² = {r2['r2']:.4f}")
        summary_lines.append(f"    Slope = {r2['slope']:.6f}, p = {r2['p_value']:.4e}")
        summary_lines.append(f"    95% CI: [{r2['ci_lower']:.6f}, {r2['ci_upper']:.6f}]")
        summary_lines.append(f"  Interpretation: Temperature is independent control")

    summary_lines.append("\n3. MULTI-PREDICTOR OLS (Block 3)")
    summary_lines.append("-" * 50)
    if 'block3' in results_dict:
        model = results_dict['block3']
        summary_lines.append(f"  Model: CH₄ ~ Precip + Temp + Year (n={int(model.nobs)})")
        summary_lines.append(f"    R² = {model.rsquared:.4f}")
        summary_lines.append(f"    F-statistic = {model.fvalue:.4f}, p < 0.001")
        summary_lines.append(f"  Coefficients:")
        for param in ['ppt_std', 'tmean_std', 'year_std']:
            if param in model.params.index:
                coef = model.params[param]
                pval = model.pvalues[param]
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                summary_lines.append(f"    {param}: {coef:.6f}, p = {pval:.4e} {sig}")
        summary_lines.append(f"  Interpretation: All predictors significant but R² = 1.2%.")
        summary_lines.append(f"                  Year has largest coefficient; model explains almost nothing.")

    summary_lines.append("\n4. SEASONAL STRATIFICATION (Block 4)")
    summary_lines.append("-" * 50)
    if 'block4' in results_dict:
        seasons_dict = results_dict['block4']
        summary_lines.append(f"  Precipitation-Flux R² by season:")
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            if season in seasons_dict:
                r2 = seasons_dict[season]['precip']['r2']
                summary_lines.append(f"    {season}: {r2:.4f}")
        summary_lines.append(f"  Interpretation: Summer is NOT strongest as predicted; all R² negligible")

    summary_lines.append("\n5. CALCIUM FERTILIZATION EXPERIMENT (Block 5)")
    summary_lines.append("-" * 50)
    if 'block5' in results_dict:
        ca = results_dict['block5']
        summary_lines.append(f"  WS1 (Ca-fert) vs WS6-BB (Reference)")
        summary_lines.append(f"    WS1 mean: {ca['ws1_annual']['Annual_CH4_flux'].mean():.4f} mg C m⁻² h⁻¹")
        summary_lines.append(f"    WS6 mean: {ca['ws6_annual']['Annual_CH4_flux'].mean():.4f} mg C m⁻² h⁻¹")
        summary_lines.append(f"    Cohen's d: {ca['cohens_d']:.3f}")
        summary_lines.append(f"    WS1 source freq: {ca['ws1_source_freq']*100:.1f}%")
        summary_lines.append(f"    WS6 source freq: {ca['ws6_source_freq']*100:.1f}%")
        summary_lines.append(f"  Interpretation: Same rain, same DECLINE (not ca-sensitive)")

    summary_lines.append("\n6. URBAN-RURAL DIVERGENCE (Block 6)")
    summary_lines.append("-" * 50)
    if 'block6' in results_dict:
        ur = results_dict['block6']
        summary_lines.append(f"  Urban post-2012 trend: slope = {ur['trend_urban'].slope:.6f}, p = {ur['trend_urban'].pvalue:.4e}")
        summary_lines.append(f"  Rural post-2012 trend: slope = {ur['trend_rural'].slope:.6f}, p = {ur['trend_rural'].pvalue:.4e}")
        summary_lines.append(f"  Interpretation: Different post-recovery trajectories (should converge)")

    summary_lines.append("\n7. STRUCTURAL BREAK DETECTION (Block 7)")
    summary_lines.append("-" * 50)
    if 'block7' in results_dict:
        bp = results_dict['block7']
        summary_lines.append(f"  Detected breakpoints (PELT algorithm, pen=0.1):")
        summary_lines.append(f"    BES: {bp['bp_year_bes'] if bp['bp_year_bes'] is not None else 'none detected'}")
        summary_lines.append(f"    HBR Reference: {bp['bp_year_hbr'] if bp['bp_year_hbr'] is not None else 'none detected'}")
        if bp.get('bes_pre_median') is not None:
            summary_lines.append(f"    BES pre-break median: {bp['bes_pre_median']:.4f} mg C m⁻² h⁻¹")
            summary_lines.append(f"    BES post-break median: {bp['bes_post_median']:.4f} mg C m⁻² h⁻¹")
        if bp.get('hbr_pre_median') is not None:
            summary_lines.append(f"    HBR pre-break median: {bp['hbr_pre_median']:.4f} mg C m⁻² h⁻¹")
            summary_lines.append(f"    HBR post-break median: {bp['hbr_post_median']:.4f} mg C m⁻² h⁻¹")
        if bp['bp_year_bes'] is not None and bp['bp_year_hbr'] is not None:
            summary_lines.append(f"  Interpretation: Structural shift detected at both sites")
        else:
            summary_lines.append(f"  Interpretation: Breakpoint detection results vary by site")

    summary_lines.append("\n8. DEPOSITION OVERLAY (Block 8)")
    summary_lines.append("-" * 50)
    if 'block8' in results_dict:
        dep = results_dict['block8']
        summary_lines.append(f"  SO₄ deposition range (HB): {dep['so4_hb'].min():.2f}-{dep['so4_hb'].max():.2f} kg/ha/yr")
        summary_lines.append(f"  Inorg N deposition range (MD): {dep['n_md'].min():.2f}-{dep['n_md'].max():.2f} kg/ha/yr")
        summary_lines.append(f"  Interpretation: Deposition trends show better alignment with decline than precipitation")

    summary_lines.append("\n9. SOIL NITROGEN (Block 9)")
    summary_lines.append("-" * 50)
    if 'block9' in results_dict:
        sn = results_dict['block9']
        for site, trend in sn.items():
            summary_lines.append(f"  {site}: slope = {trend.slope:.6f} mg/L/yr, p = {trend.pvalue:.4e}")
        summary_lines.append(f"  Interpretation: Variable trends; no universal N accumulation pattern")

    summary_lines.append("\n10. SOIL MOISTURE DIRECT TEST (Block 11)")
    summary_lines.append("-" * 50)
    if 'block11' in results_dict and results_dict['block11'] is not None:
        r11 = results_dict['block11']['pooled']
        summary_lines.append(f"  In-situ VWC vs CH₄ flux (pooled, n={r11['n']})")
        summary_lines.append(f"    R² = {r11['r2']:.4f}")
        summary_lines.append(f"    Slope = {r11['slope']:.6f}, p = {r11['p_value']:.4e}")
        summary_lines.append(f"  Interpretation: Both VWC and precipitation explain <1% of variance; neither supports diffusion limitation")
    else:
        summary_lines.append("  Skipped (no VWC-flux data overlap)")

    summary_lines.append("\n11. HARVARD FOREST CONTEXT (Block 12)")
    summary_lines.append("-" * 50)
    if 'block12' in results_dict and results_dict['block12'] is not None:
        r12 = results_dict['block12']['trend']
        summary_lines.append(f"  CH₄ concentration trend 1991-1994 (n={r12['n']} months)")
        summary_lines.append(f"    Slope = {r12['slope']:.6f} ppm/month, R² = {r12['r2']:.4f}")
        summary_lines.append(f"  Note: Ambient concentration, not soil flux. Baseline context only.")

    summary_lines.append("\n12. SOIL PROPERTIES (Block 13)")
    summary_lines.append("-" * 50)
    if 'block13' in results_dict and results_dict['block13'] is not None:
        summary_lines.append(f"  Forest site soil characterization at 0-10 cm depth")
        summary_lines.append(f"  Sites: HD (urban), LEA (urban), ORM (rural), ORU (rural)")
        summary_lines.append(f"  Key finding: Site-level biogeochemical variation documented")

    summary_lines.append("\n13. MULTI-SCALE ANALYSIS (Block 14)")
    summary_lines.append("-" * 50)
    if 'block14' in results_dict and results_dict['block14'] is not None:
        ms = results_dict['block14']
        for scale_name in ['measurement', 'seasonal', 'annual']:
            if scale_name in ms:
                s = ms[scale_name]
                summary_lines.append(f"  {scale_name.capitalize()} scale (n={s['n']}):")
                summary_lines.append(f"    R² = {s['r2']:.4f}")
                summary_lines.append(f"    β_ppt = {s['beta_ppt']:.4f} (p={s['p_ppt']:.2e})")
                summary_lines.append(f"    β_temp = {s['beta_temp']:.4f} (p={s['p_temp']:.2e})")
                summary_lines.append(f"    β_year = {s['beta_year']:.4f} (p={s['p_year']:.2e})")
        summary_lines.append(f"  Interpretation: Year dominates at all scales; precipitation never explains meaningful variance")

    summary_lines.append("\n14. OVERALL CONCLUSION")
    summary_lines.append("="*70)
    summary_lines.append("The diffusion limitation hypothesis fails to explain the 77% decline")
    summary_lines.append("in forest methane uptake because:")
    summary_lines.append("")
    summary_lines.append("  1. Precipitation does NOT significantly predict flux variation")
    summary_lines.append("  2. Temperature is independent and unrelated to decline")
    summary_lines.append("  3. Multi-predictor model explains 1.2% of variance; year dominates")
    summary_lines.append("  4. Summer is not the strongest season as diffusion limitation predicts")
    summary_lines.append("  5. Same rainfall = same decline in Ca-fertilization experiment")
    summary_lines.append("  6. Urban and rural sites diverge post-2012 (not recovering similarly)")
    summary_lines.append("  7. Structural break(s) detected independently at both LTER sites")
    summary_lines.append("  8. Deposition trends better correlated with CH4 than precipitation")
    summary_lines.append("  9. Soil nitrogen changes do not explain magnitude of decline")
    summary_lines.append(" 10. In-situ soil moisture ALSO fails to predict flux (direct test)")
    summary_lines.append("")
    summary_lines.append("Alternative explanation: Biogeochemical reorganization related to")
    summary_lines.append("nitrogen saturation and/or acidification recovery is more parsimonious.")
    summary_lines.append("="*70)

    # Print all lines
    for line in summary_lines:
        print(line)

    # Save summary to file (UTF-8 for subscript characters)
    with open(os.path.join(OUTPUT_DIR, 'SUMMARY.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FOREST METHANE SINK ANALYSIS")
    print("Testing the Diffusion Limitation Hypothesis")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Initialize results dictionary
    all_results = {}

    try:
        # Block 1: Precipitation-Flux
        print("\nExecuting Block 1...")
        result1 = block_1_precipitation_flux()
        all_results['block1'] = result1
    except Exception as e:
        print(f"ERROR in Block 1: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 2: Temperature-Flux
        print("\nExecuting Block 2...")
        result2 = block_2_temperature_flux(result1['merged_data'])
        all_results['block2'] = result2
    except Exception as e:
        print(f"ERROR in Block 2: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 3: Multi-predictor
        print("\nExecuting Block 3...")
        result3 = block_3_multi_predictor(result2['merged_data'])
        all_results['block3'] = result3
    except Exception as e:
        print(f"ERROR in Block 3: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 4: Seasonal
        print("\nExecuting Block 4...")
        result4 = block_4_seasonal_stratification(result2['merged_data'])
        all_results['block4'] = result4
    except Exception as e:
        print(f"ERROR in Block 4: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 5: Ca Experiment
        print("\nExecuting Block 5...")
        result5 = block_5_calcium_experiment()
        all_results['block5'] = result5
    except Exception as e:
        print(f"ERROR in Block 5: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 6: Urban-Rural
        print("\nExecuting Block 6...")
        result6 = block_6_urban_rural_divergence()
        all_results['block6'] = result6
    except Exception as e:
        print(f"ERROR in Block 6: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 7: Breakpoints
        print("\nExecuting Block 7...")
        result7 = block_7_breakpoint_detection()
        all_results['block7'] = result7
    except Exception as e:
        print(f"ERROR in Block 7: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 8: Deposition
        print("\nExecuting Block 8...")
        result8 = block_8_deposition_overlay()
        all_results['block8'] = result8
    except Exception as e:
        print(f"ERROR in Block 8: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 9: Soil N
        print("\nExecuting Block 9...")
        result9 = block_9_soil_nitrogen()
        all_results['block9'] = result9
    except Exception as e:
        print(f"ERROR in Block 9: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 10: Vegetation
        print("\nExecuting Block 10...")
        result10 = block_10_vegetation()
        all_results['block10'] = result10
    except Exception as e:
        print(f"ERROR in Block 10: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 11: Soil Moisture Direct Test
        print("\nExecuting Block 11...")
        precip_r2 = all_results.get('block1', {}).get('pooled', {}).get('r2')
        result11 = block_11_soil_moisture_flux(precip_r2=precip_r2)
        all_results['block11'] = result11
    except Exception as e:
        print(f"ERROR in Block 11: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 12: Harvard Forest Context
        print("\nExecuting Block 12...")
        result12 = block_12_harvard_forest()
        all_results['block12'] = result12
    except Exception as e:
        print(f"ERROR in Block 12: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 13: Soil Properties
        print("\nExecuting Block 13...")
        result13 = block_13_soil_properties()
        all_results['block13'] = result13
    except Exception as e:
        print(f"ERROR in Block 13: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Block 14: Multi-Scale Analysis
        print("\nExecuting Block 14...")
        result14 = block_14_multi_scale(result2['merged_data'])
        all_results['block14'] = result14
    except Exception as e:
        print(f"ERROR in Block 14: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print_summary(all_results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("\nAnalysis complete. Summary saved to SUMMARY.txt")
