#!/usr/bin/env python3
"""
Supplemental Robustness Checks
==============================

Four analyses requested during peer review:

1. Linear Mixed-Effects Model (LMM)
   CH4 ~ Precip + Temp + Year + (1 | Site)
   Addresses pseudoreplication from repeated chamber measurements
   at fixed sites (Referee 1, Comment 2).

2. Outlier Sensitivity Test
   Re-runs key regressions (Predictions 1, 5) WITHOUT the ±3 SD
   per-site-year filter, retaining only hotspot-site exclusion.
   Addresses concern that trimming removes extreme events where
   diffusion limitation physically manifests (Referee 1, Comment 4).

3. Quadratic (Non-Linear) Moisture-Flux Test
   Fits Flux ~ Precip + Precip² and Flux ~ VWC + VWC² to test
   whether the unimodal moisture-flux relationship expected from
   diffusion theory is masked by the linear specification.
   Addresses Referee 1, Comment 1.

4. Pre-Breakpoint Precipitation Regression
   Runs precip-flux regression on pre-2002 BES data only, when the
   methanotrophic community was presumably intact.
   Addresses Referee 1, Comment 3.

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
# ANALYSIS 3: QUADRATIC (NON-LINEAR) MOISTURE-FLUX TEST
# ============================================================================

def load_soil_moisture():
    """Load BES soil moisture (VWC) data — mirrors master_analysis.py."""
    fpath = os.path.join(DATA_DIR, "knb-lter-bes.3400/BES_TempVWC_2011-2020.csv")
    df = pd.read_csv(fpath)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['YearMonth'] = df['Timestamp'].dt.to_period('M')

    # Site code column
    if 'Site_Code' not in df.columns:
        for col in ['Site', 'site', 'SiteCode', 'site_code']:
            if col in df.columns:
                df = df.rename(columns={col: 'Site_Code'})
                break

    # Extract VWC columns
    vwc_cols = [col for col in df.columns if col.startswith('Port_') and col.endswith('_VWC')]
    if not vwc_cols:
        vwc_cols = [col for col in df.columns if 'VWC' in col and col != 'Site_Code']

    df['Mean_VWC'] = df[vwc_cols].mean(axis=1)
    df = df[['Site_Code', 'Timestamp', 'Year', 'Month', 'YearMonth', 'Mean_VWC']].copy()
    df = df.dropna(subset=['Mean_VWC'])

    print(f"  Loaded soil moisture (VWC): {len(df)} rows, sites: {sorted(df['Site_Code'].unique())}")
    return df


def run_quadratic_test(bes_raw):
    """
    Test whether a quadratic (unimodal) moisture-flux model improves on linear.
    Fits: Flux ~ Precip + Precip² and Flux ~ VWC + VWC²
    If the quadratic term is non-significant and R² remains negligible,
    the linear falsification holds even for non-linear relationships.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: QUADRATIC (NON-LINEAR) MOISTURE-FLUX TEST")
    print("=" * 70)

    results = {}

    # --- 3A: Precipitation quadratic ---
    print("\n  [A] Precipitation: Flux ~ Precip + Precip²")
    df = prepare_merged_dataset(bes_raw, apply_sd_filter=True)
    df_clean = df.dropna(subset=['CH4_flux', 'ppt_mm']).copy()

    # Linear
    slope_l, intercept_l, r_l, p_l, se_l = stats.linregress(df_clean['ppt_mm'], df_clean['CH4_flux'])
    r2_linear = r_l ** 2

    # Quadratic
    df_clean['ppt_sq'] = df_clean['ppt_mm'] ** 2
    X_quad = sm.add_constant(df_clean[['ppt_mm', 'ppt_sq']])
    quad_model = sm.OLS(df_clean['CH4_flux'], X_quad).fit()

    r2_quad = quad_model.rsquared
    ppt_sq_beta = quad_model.params['ppt_sq']
    ppt_sq_p = quad_model.pvalues['ppt_sq']

    # F-test: quadratic vs linear
    # Compare nested models via partial F-test
    n = len(df_clean)
    f_stat = ((r2_quad - r2_linear) / 1) / ((1 - r2_quad) / (n - 3))
    from scipy.stats import f as f_dist
    f_p = 1 - f_dist.cdf(f_stat, 1, n - 3)

    print(f"    n = {n}")
    print(f"    Linear R² = {r2_linear:.6f}")
    print(f"    Quadratic R² = {r2_quad:.6f}")
    print(f"    Precip² coefficient: β = {ppt_sq_beta:.2e}, p = {ppt_sq_p:.2e}")
    print(f"    Partial F-test (quad vs linear): F = {f_stat:.4f}, p = {f_p:.4f}")
    print(f"    R² improvement: {(r2_quad - r2_linear)*100:.4f} percentage points")

    results['precip_linear_r2'] = r2_linear
    results['precip_quad_r2'] = r2_quad
    results['precip_sq_beta'] = ppt_sq_beta
    results['precip_sq_p'] = ppt_sq_p
    results['precip_f_stat'] = f_stat
    results['precip_f_p'] = f_p
    results['precip_n'] = n

    # --- 3B: VWC quadratic ---
    print("\n  [B] VWC: Flux ~ VWC + VWC²")
    try:
        vwc = load_soil_moisture()
        vwc_monthly = vwc.groupby(['Site_Code', 'YearMonth']).agg({'Mean_VWC': 'mean'}).reset_index()

        bes_forest = bes_raw[bes_raw['Site'].isin(FOREST_SITES) & bes_raw['CH4_flux'].notna()].copy()
        bes_forest = trim_outliers(bes_forest, apply_sd_filter=True)

        all_vwc = []
        for site in FOREST_SITES:
            site_flux = bes_forest[bes_forest['Site'] == site].copy()
            site_vwc = vwc_monthly[vwc_monthly['Site_Code'] == site].copy()
            if len(site_flux) == 0 or len(site_vwc) == 0:
                continue
            merged = pd.merge(
                site_flux, site_vwc,
                left_on=['Site', 'YearMonth'], right_on=['Site_Code', 'YearMonth'],
                how='inner'
            )
            all_vwc.append(merged)

        if all_vwc:
            df_vwc = pd.concat(all_vwc, ignore_index=True)
            df_vwc = df_vwc.dropna(subset=['CH4_flux', 'Mean_VWC'])

            # Linear
            slope_vl, intercept_vl, r_vl, p_vl, se_vl = stats.linregress(df_vwc['Mean_VWC'], df_vwc['CH4_flux'])
            r2_vwc_linear = r_vl ** 2

            # Quadratic
            df_vwc['vwc_sq'] = df_vwc['Mean_VWC'] ** 2
            X_vwc_quad = sm.add_constant(df_vwc[['Mean_VWC', 'vwc_sq']])
            vwc_quad_model = sm.OLS(df_vwc['CH4_flux'], X_vwc_quad).fit()

            r2_vwc_quad = vwc_quad_model.rsquared
            vwc_sq_beta = vwc_quad_model.params['vwc_sq']
            vwc_sq_p = vwc_quad_model.pvalues['vwc_sq']

            n_vwc = len(df_vwc)
            f_stat_vwc = ((r2_vwc_quad - r2_vwc_linear) / 1) / ((1 - r2_vwc_quad) / (n_vwc - 3))
            f_p_vwc = 1 - f_dist.cdf(f_stat_vwc, 1, n_vwc - 3)

            print(f"    n = {n_vwc}")
            print(f"    Linear R² = {r2_vwc_linear:.6f}")
            print(f"    Quadratic R² = {r2_vwc_quad:.6f}")
            print(f"    VWC² coefficient: β = {vwc_sq_beta:.2e}, p = {vwc_sq_p:.2e}")
            print(f"    Partial F-test (quad vs linear): F = {f_stat_vwc:.4f}, p = {f_p_vwc:.4f}")
            print(f"    R² improvement: {(r2_vwc_quad - r2_vwc_linear)*100:.4f} percentage points")

            results['vwc_linear_r2'] = r2_vwc_linear
            results['vwc_quad_r2'] = r2_vwc_quad
            results['vwc_sq_beta'] = vwc_sq_beta
            results['vwc_sq_p'] = vwc_sq_p
            results['vwc_f_stat'] = f_stat_vwc
            results['vwc_f_p'] = f_p_vwc
            results['vwc_n'] = n_vwc
        else:
            print("    No VWC-flux overlap data available.")
            results['vwc_n'] = 0
    except Exception as e:
        print(f"    VWC test failed: {e}")
        results['vwc_n'] = 0

    return results


# ============================================================================
# ANALYSIS 4: PRE-BREAKPOINT PRECIPITATION REGRESSION
# ============================================================================

def run_pre_breakpoint_test(bes_raw):
    """
    Run precip-flux regression on pre-2002 BES data only.
    If diffusion was the primary driver before biological collapse,
    R² should be higher in 1998-2002 than in the full record.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: PRE-BREAKPOINT PRECIPITATION REGRESSION")
    print("=" * 70)

    results = {}

    df_full = prepare_merged_dataset(bes_raw, apply_sd_filter=True)
    df_full_clean = df_full.dropna(subset=['CH4_flux', 'ppt_mm']).copy()

    # Full record
    slope_f, intercept_f, r_f, p_f, se_f = stats.linregress(df_full_clean['ppt_mm'], df_full_clean['CH4_flux'])
    r2_full = r_f ** 2
    n_full = len(df_full_clean)

    print(f"\n  Full record (1998-2025):")
    print(f"    n = {n_full}, R² = {r2_full:.6f}, slope = {slope_f:.6f}, p = {p_f:.2e}")

    results['full_n'] = n_full
    results['full_r2'] = r2_full
    results['full_slope'] = slope_f
    results['full_p'] = p_f

    # Pre-2002 (breakpoint year)
    df_pre = df_full_clean[df_full_clean['Year'] <= 2002].copy()
    n_pre = len(df_pre)

    if n_pre > 10:
        slope_pre, intercept_pre, r_pre, p_pre, se_pre = stats.linregress(df_pre['ppt_mm'], df_pre['CH4_flux'])
        r2_pre = r_pre ** 2

        print(f"\n  Pre-breakpoint (1998-2002):")
        print(f"    n = {n_pre}, R² = {r2_pre:.6f}, slope = {slope_pre:.6f}, p = {p_pre:.2e}")

        results['pre_n'] = n_pre
        results['pre_r2'] = r2_pre
        results['pre_slope'] = slope_pre
        results['pre_p'] = p_pre
    else:
        print(f"\n  Pre-breakpoint: insufficient data (n = {n_pre})")
        results['pre_n'] = n_pre
        results['pre_r2'] = np.nan

    # Post-2002
    df_post = df_full_clean[df_full_clean['Year'] > 2002].copy()
    n_post = len(df_post)

    if n_post > 10:
        slope_post, intercept_post, r_post, p_post, se_post = stats.linregress(df_post['ppt_mm'], df_post['CH4_flux'])
        r2_post = r_post ** 2

        print(f"\n  Post-breakpoint (2003-2025):")
        print(f"    n = {n_post}, R² = {r2_post:.6f}, slope = {slope_post:.6f}, p = {p_post:.2e}")

        results['post_n'] = n_post
        results['post_r2'] = r2_post
        results['post_slope'] = slope_post
        results['post_p'] = p_post
    else:
        print(f"\n  Post-breakpoint: insufficient data (n = {n_post})")
        results['post_n'] = n_post
        results['post_r2'] = np.nan

    # Also try quadratic on pre-2002
    if n_pre > 10:
        df_pre['ppt_sq'] = df_pre['ppt_mm'] ** 2
        X_pre_quad = sm.add_constant(df_pre[['ppt_mm', 'ppt_sq']])
        pre_quad_model = sm.OLS(df_pre['CH4_flux'], X_pre_quad).fit()
        r2_pre_quad = pre_quad_model.rsquared
        pre_sq_p = pre_quad_model.pvalues['ppt_sq']

        print(f"\n  Pre-breakpoint quadratic:")
        print(f"    R² = {r2_pre_quad:.6f}, Precip² p = {pre_sq_p:.2e}")

        results['pre_quad_r2'] = r2_pre_quad
        results['pre_quad_sq_p'] = pre_sq_p

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

    # Analysis 3: Quadratic test
    quad_results = run_quadratic_test(bes_raw)

    # Analysis 4: Pre-breakpoint regression
    pre_results = run_pre_breakpoint_test(bes_raw)

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

    lines.append("\n3. QUADRATIC (NON-LINEAR) MOISTURE-FLUX TEST")
    lines.append("-" * 50)
    lines.append(f"  [A] Precipitation (n = {quad_results['precip_n']})")
    lines.append(f"    Linear R² = {quad_results['precip_linear_r2']:.6f}")
    lines.append(f"    Quadratic R² = {quad_results['precip_quad_r2']:.6f}")
    lines.append(f"    Precip² coefficient: β = {quad_results['precip_sq_beta']:.2e}, p = {quad_results['precip_sq_p']:.2e}")
    lines.append(f"    F-test (quad vs linear): F = {quad_results['precip_f_stat']:.4f}, p = {quad_results['precip_f_p']:.4f}")
    if quad_results.get('vwc_n', 0) > 0:
        lines.append(f"  [B] VWC (n = {quad_results['vwc_n']})")
        lines.append(f"    Linear R² = {quad_results['vwc_linear_r2']:.6f}")
        lines.append(f"    Quadratic R² = {quad_results['vwc_quad_r2']:.6f}")
        lines.append(f"    VWC² coefficient: β = {quad_results['vwc_sq_beta']:.2e}, p = {quad_results['vwc_sq_p']:.2e}")
        lines.append(f"    F-test (quad vs linear): F = {quad_results['vwc_f_stat']:.4f}, p = {quad_results['vwc_f_p']:.4f}")
    lines.append("")
    lines.append("  Interpretation: If the true moisture-flux relationship is unimodal,")
    lines.append("  the quadratic term should be significant and R² should increase")
    lines.append("  substantially over the linear model.")

    lines.append("\n4. PRE-BREAKPOINT PRECIPITATION REGRESSION")
    lines.append("-" * 50)
    lines.append(f"  Full record: n = {pre_results['full_n']}, R² = {pre_results['full_r2']:.6f}, p = {pre_results['full_p']:.2e}")
    if not np.isnan(pre_results.get('pre_r2', np.nan)):
        lines.append(f"  Pre-2002:    n = {pre_results['pre_n']}, R² = {pre_results['pre_r2']:.6f}, p = {pre_results['pre_p']:.2e}")
    if not np.isnan(pre_results.get('post_r2', np.nan)):
        lines.append(f"  Post-2002:   n = {pre_results['post_n']}, R² = {pre_results['post_r2']:.6f}, p = {pre_results['post_p']:.2e}")
    if 'pre_quad_r2' in pre_results:
        lines.append(f"  Pre-2002 quadratic: R² = {pre_results['pre_quad_r2']:.6f}, Precip² p = {pre_results['pre_quad_sq_p']:.2e}")
    lines.append("")
    lines.append("  Interpretation: If diffusion controlled the sink before biological")
    lines.append("  collapse, pre-breakpoint R² should be substantially higher than the")
    lines.append("  full-record R². If it is also negligible, diffusion was never the")
    lines.append("  dominant control at these sites.")

    lines.append("\n" + "=" * 70)

    summary_text = '\n'.join(lines)
    print("\n" + summary_text)

    with open(os.path.join(OUTPUT_DIR, 'SUPPLEMENTAL_RESULTS.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\nResults saved to output/SUPPLEMENTAL_RESULTS.txt")


if __name__ == '__main__':
    main()
