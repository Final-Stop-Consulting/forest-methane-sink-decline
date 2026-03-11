#!/usr/bin/env python3
"""
Supplemental Robustness Checks
==============================

Two analyses requested during peer review:

1. Linear Mixed-Effects Model (LMM)
   CH4 ~ Precip + Temp + Year + (1 | Site)
   Addresses pseudoreplication from repeated chamber measurements
   at fixed sites (Referee 2, Comment 2).

2. Outlier Sensitivity Test
   Re-runs key regressions (Predictions 1, 5) WITHOUT the ±3 SD
   per-site-year filter, retaining only hotspot-site exclusion.
   Addresses concern that trimming removes extreme events where
   diffusion limitation physically manifests (Referee 2, Comment 4).

Usage:
    cd Analysis
    python supplemental_robustness.py

Requires the same data files as master_analysis.py (see ../Data/README.md).
Output: SUPPLEMENTAL_RESULTS.txt in output/

Author:  Victor Edmonds
Contact: victoredmonds@gmail.com
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION (mirrors master_analysis.py)
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

DATA_DIR = os.path.join(_PROJECT_DIR, "Data")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

URBAN_SITES = ['HD', 'LEA', 'MCD', 'GB', 'GLY', 'UMBC']
RURAL_SITES = ['ORM', 'ORU', 'ORLR', 'ORUR', 'CAH']
FOREST_SITES = URBAN_SITES + RURAL_SITES
HOTSPOT_SITES = ['GB', 'ORLR']
OUTLIER_SD_THRESHOLD = 3
MISSING_VALUES = [-9999.99, -9999, -9, -99.999]

PRISM_MAPPING = {
    'HD': 'BES', 'LEA': 'BES', 'MCD': 'BES',
    'GB': 'BES', 'GLY': 'BES', 'UMBC': 'BES',
    'ORM': 'OregonRidge', 'ORU': 'OregonRidge',
    'ORLR': 'OregonRidge', 'ORUR': 'OregonRidge',
    'CAH': 'OregonRidge',
}

# ============================================================================
# DATA LOADING (from master_analysis.py)
# ============================================================================

def load_bes_flux():
    fpath = os.path.join(DATA_DIR, "BES_trace-gas-collection_1998_2025.csv")
    df = pd.read_csv(fpath)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Year'].astype(int)
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df['CH4_flux'] = df['CH4_flux'].replace(MISSING_VALUES, np.nan)
    return df

def load_prism(site_key):
    prism_site = PRISM_MAPPING.get(site_key, 'BES')
    if prism_site == 'BES':
        fpath = os.path.join(DATA_DIR, "PRISM/BES-PRISM_ppt_tmean_stable_4km_199801_202506_39.3400_-76.6200.csv")
    elif prism_site == 'OregonRidge':
        fpath = os.path.join(DATA_DIR, "PRISM/OregonRidge-PRISM_ppt_tmean_stable_4km_199801_202506_39.4970_-76.6890.csv")
    else:
        return None
    df = pd.read_csv(fpath, skiprows=10)
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')
    if 'ppt (mm)' in df.columns:
        df = df.rename(columns={'ppt (mm)': 'ppt_mm', 'tmean (degrees C)': 'tmean_c'})
    return df

def trim_outliers(df, apply_sd_filter=True):
    """Remove hotspot sites. Optionally apply ±3 SD per-site-year trim."""
    df_out = df[~df['Site'].isin(HOTSPOT_SITES)].copy()
    n_hotspot = len(df) - len(df_out)

    if not apply_sd_filter:
        print(f"  Hotspot exclusion only: removed {n_hotspot} rows ({len(df_out)} remaining)")
        return df_out.reset_index(drop=True)

    def _trim_group(g):
        mu = g['CH4_flux'].mean()
        sd = g['CH4_flux'].std()
        if sd == 0 or np.isnan(sd):
            return g
        return g[np.abs(g['CH4_flux'] - mu) <= OUTLIER_SD_THRESHOLD * sd]

    df_trimmed = df_out.groupby(['Site', 'Year'], group_keys=False).apply(_trim_group)
    n_sd = len(df_out) - len(df_trimmed)
    print(f"  Full filter: removed {n_hotspot} hotspot + {n_sd} SD-trim rows ({len(df_trimmed)} remaining)")
    return df_trimmed.reset_index(drop=True)

def prepare_merged_dataset(bes, apply_sd_filter=True):
    """Filter BES data and merge with PRISM climate."""
    forest_mask = bes['Site'].isin(FOREST_SITES)
    df = bes[forest_mask & bes['CH4_flux'].notna()].copy()
    df = trim_outliers(df, apply_sd_filter=apply_sd_filter)

    all_data = []
    for site in FOREST_SITES:
        site_data = df[df['Site'] == site].copy()
        if len(site_data) == 0:
            continue
        prism = load_prism(site)
        if prism is None:
            continue
        merged = pd.merge(
            site_data, prism[['YearMonth', 'ppt_mm', 'tmean_c']],
            on='YearMonth', how='inner'
        )
        all_data.append(merged)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# ============================================================================
# ANALYSIS 1: LINEAR MIXED-EFFECTS MODEL
# ============================================================================

def run_lmm(df):
    """
    Fit CH4 ~ Precip + Temp + Year + (1 | Site)
    using the standard-filtered dataset (n ≈ 9,359).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: LINEAR MIXED-EFFECTS MODEL")
    print("=" * 70)

    df_model = df.dropna(subset=['CH4_flux', 'ppt_mm', 'tmean_c', 'Year', 'Site']).copy()

    # Standardize predictors (same as master_analysis Block 3)
    df_model['ppt_std'] = (df_model['ppt_mm'] - df_model['ppt_mm'].mean()) / df_model['ppt_mm'].std()
    df_model['tmean_std'] = (df_model['tmean_c'] - df_model['tmean_c'].mean()) / df_model['tmean_c'].std()
    df_model['year_std'] = (df_model['Year'] - df_model['Year'].mean()) / df_model['Year'].std()

    print(f"\n  Dataset: n = {len(df_model)}, sites = {df_model['Site'].nunique()}")
    print(f"  Sites: {sorted(df_model['Site'].unique())}")

    # --- OLS (for comparison) ---
    X_ols = sm.add_constant(df_model[['ppt_std', 'tmean_std', 'year_std']])
    ols_model = sm.OLS(df_model['CH4_flux'], X_ols).fit()

    print(f"\n  OLS (baseline, for comparison):")
    print(f"    R² = {ols_model.rsquared:.4f}")
    for param in ['ppt_std', 'tmean_std', 'year_std']:
        print(f"    {param}: β = {ols_model.params[param]:.4f}, p = {ols_model.pvalues[param]:.2e}")

    # --- LMM: random intercept for Site ---
    exog = df_model[['ppt_std', 'tmean_std', 'year_std']].copy()
    exog.insert(0, 'Intercept', 1.0)

    lmm = MixedLM(
        endog=df_model['CH4_flux'],
        exog=exog,
        groups=df_model['Site']
    )
    lmm_result = lmm.fit(reml=True)

    print(f"\n  LMM: CH4 ~ Precip + Temp + Year + (1 | Site)")
    print(f"    Converged: {lmm_result.converged}")
    print(f"    Log-likelihood: {lmm_result.llf:.2f}")
    print(f"    Site variance (random intercept): {lmm_result.cov_re.iloc[0, 0]:.4f}")
    print(f"    Residual variance: {lmm_result.scale:.4f}")

    print(f"\n  Fixed effects:")
    for param in ['ppt_std', 'tmean_std', 'year_std']:
        beta = lmm_result.fe_params[param]
        pval = lmm_result.pvalues[param]
        se = lmm_result.bse[param]
        print(f"    {param}: β = {beta:.4f} (SE = {se:.4f}), p = {pval:.2e}")

    # Summary comparison
    print(f"\n  Comparison (OLS → LMM):")
    for param in ['ppt_std', 'tmean_std', 'year_std']:
        ols_b = ols_model.params[param]
        ols_p = ols_model.pvalues[param]
        lmm_b = lmm_result.fe_params[param]
        lmm_p = lmm_result.pvalues[param]
        print(f"    {param}: β {ols_b:.4f} → {lmm_b:.4f}, p {ols_p:.2e} → {lmm_p:.2e}")

    return ols_model, lmm_result


# ============================================================================
# ANALYSIS 2: OUTLIER SENSITIVITY TEST
# ============================================================================

def run_outlier_sensitivity(bes_raw):
    """
    Re-run Prediction 1 (precip-flux) and Prediction 5 (multi-predictor)
    with hotspot site exclusion only (no ±3 SD trim).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: OUTLIER SENSITIVITY TEST")
    print("=" * 70)

    # --- Standard filtered dataset ---
    print("\n  [A] Standard filter (hotspot + ±3 SD):")
    df_std = prepare_merged_dataset(bes_raw, apply_sd_filter=True)

    # --- Hotspot-only dataset (no SD trim) ---
    print("\n  [B] Hotspot exclusion only (no SD trim):")
    df_no_sd = prepare_merged_dataset(bes_raw, apply_sd_filter=False)

    results = {}

    for label, df in [("Standard filter", df_std), ("No SD trim", df_no_sd)]:
        print(f"\n  --- {label} (n = {len(df)}) ---")

        df_clean = df.dropna(subset=['CH4_flux', 'ppt_mm', 'tmean_c']).copy()

        # Prediction 1: Precip-flux regression
        slope, intercept, r, p, se = stats.linregress(df_clean['ppt_mm'], df_clean['CH4_flux'])
        r2 = r ** 2
        print(f"    Prediction 1 (Precip ~ Flux): R² = {r2:.4f}, slope = {slope:.6f}, p = {p:.2e}")

        # Prediction 5: Multi-predictor OLS
        df_clean['ppt_std'] = (df_clean['ppt_mm'] - df_clean['ppt_mm'].mean()) / df_clean['ppt_mm'].std()
        df_clean['tmean_std'] = (df_clean['tmean_c'] - df_clean['tmean_c'].mean()) / df_clean['tmean_c'].std()
        df_clean['year_std'] = (df_clean['Year'] - df_clean['Year'].mean()) / df_clean['Year'].std()

        X = sm.add_constant(df_clean[['ppt_std', 'tmean_std', 'year_std']])
        model = sm.OLS(df_clean['CH4_flux'], X).fit()

        print(f"    Prediction 5 (Multi-predictor): R² = {model.rsquared:.4f}")
        for param in ['ppt_std', 'tmean_std', 'year_std']:
            print(f"      {param}: β = {model.params[param]:.4f}, p = {model.pvalues[param]:.2e}")

        results[label] = {
            'n': len(df_clean),
            'precip_r2': r2,
            'precip_slope': slope,
            'precip_p': p,
            'multi_r2': model.rsquared,
            'multi_params': {p: (model.params[p], model.pvalues[p]) for p in ['ppt_std', 'tmean_std', 'year_std']},
        }

    # Comparison
    print(f"\n  --- Comparison ---")
    std = results["Standard filter"]
    nos = results["No SD trim"]
    print(f"    Sample size: {std['n']} → {nos['n']} (+{nos['n'] - std['n']} recovered)")
    print(f"    Precip R²: {std['precip_r2']:.4f} → {nos['precip_r2']:.4f}")
    print(f"    Multi R²: {std['multi_r2']:.4f} → {nos['multi_r2']:.4f}")
    for param in ['ppt_std', 'tmean_std', 'year_std']:
        sb, sp = std['multi_params'][param]
        nb, np_ = nos['multi_params'][param]
        print(f"    {param}: β {sb:.4f} → {nb:.4f}, p {sp:.2e} → {np_:.2e}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("SUPPLEMENTAL ROBUSTNESS CHECKS")
    print("=" * 70)

    # Load raw BES data
    bes_raw = load_bes_flux()

    # Prepare standard-filtered merged dataset for LMM
    print("\nPreparing standard-filtered dataset for LMM...")
    df_std = prepare_merged_dataset(bes_raw, apply_sd_filter=True)

    # Analysis 1: LMM
    ols_model, lmm_result = run_lmm(df_std)

    # Analysis 2: Outlier sensitivity
    outlier_results = run_outlier_sensitivity(bes_raw)

    # ========================================================================
    # Write summary
    # ========================================================================
    lines = []
    lines.append("SUPPLEMENTAL ROBUSTNESS CHECKS")
    lines.append("=" * 70)

    lines.append("\n1. LINEAR MIXED-EFFECTS MODEL")
    lines.append("-" * 50)
    lines.append("  Model: CH4 ~ Precip + Temp + Year + (1 | Site)")
    lines.append(f"  Dataset: n = {int(ols_model.nobs)}, REML estimation")
    lines.append(f"  Site variance (random intercept): {lmm_result.cov_re.iloc[0, 0]:.4f}")
    lines.append(f"  Residual variance: {lmm_result.scale:.4f}")
    lines.append("")
    lines.append("  Fixed effects (OLS → LMM):")
    for param in ['ppt_std', 'tmean_std', 'year_std']:
        ols_b = ols_model.params[param]
        ols_p = ols_model.pvalues[param]
        lmm_b = lmm_result.fe_params[param]
        lmm_p = lmm_result.pvalues[param]
        lines.append(f"    {param}: β = {ols_b:.4f} (p={ols_p:.2e}) → β = {lmm_b:.4f} (p={lmm_p:.2e})")
    lines.append("")
    lines.append("  Interpretation: Accounting for site-level clustering via random")
    lines.append("  intercepts. Compare coefficient magnitudes and significance to OLS.")

    lines.append("\n2. OUTLIER SENSITIVITY TEST")
    lines.append("-" * 50)
    std = outlier_results["Standard filter"]
    nos = outlier_results["No SD trim"]
    lines.append(f"  Standard filter (hotspot + ±3 SD): n = {std['n']}")
    lines.append(f"  Hotspot exclusion only (no SD trim): n = {nos['n']}")
    lines.append(f"  Measurements recovered: {nos['n'] - std['n']}")
    lines.append("")
    lines.append("  Prediction 1 (Precip ~ Flux):")
    lines.append(f"    Standard: R² = {std['precip_r2']:.4f}, slope = {std['precip_slope']:.6f}, p = {std['precip_p']:.2e}")
    lines.append(f"    No SD trim: R² = {nos['precip_r2']:.4f}, slope = {nos['precip_slope']:.6f}, p = {nos['precip_p']:.2e}")
    lines.append("")
    lines.append("  Prediction 5 (Multi-predictor OLS):")
    lines.append(f"    Standard: R² = {std['multi_r2']:.4f}")
    lines.append(f"    No SD trim: R² = {nos['multi_r2']:.4f}")
    for param in ['ppt_std', 'tmean_std', 'year_std']:
        sb, sp = std['multi_params'][param]
        nb, np_ = nos['multi_params'][param]
        lines.append(f"    {param}: β = {sb:.4f} (p={sp:.2e}) → β = {nb:.4f} (p={np_:.2e})")
    lines.append("")
    lines.append("  Interpretation: Removing the ±3 SD filter recovers extreme positive")
    lines.append("  measurements. Compare whether precipitation gains explanatory power")
    lines.append("  when these diffusion-relevant extreme events are retained.")

    lines.append("\n" + "=" * 70)

    summary_text = '\n'.join(lines)
    print("\n" + summary_text)

    with open(os.path.join(OUTPUT_DIR, 'SUPPLEMENTAL_RESULTS.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\nResults saved to output/SUPPLEMENTAL_RESULTS.txt")


if __name__ == '__main__':
    main()
