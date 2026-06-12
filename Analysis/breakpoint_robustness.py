#!/usr/bin/env python3
"""
Breakpoint Robustness Checks (Referee #2, major comment 4)
==========================================================

Four analyses characterizing the uncertainty of the PELT structural breaks
reported in master_analysis.py (BES 2002; HBR WS6-BB 2011), all on the SAME
annual-median series and l2 cost used for Figure 4 / Table 4:

  1. Block-bootstrap confidence intervals on breakpoint timing.
     Residual moving-block bootstrap around the fitted piecewise-constant model
     (preserves the calendar-year axis), B = 2000, seeded. Reports the 95%
     percentile interval on the breakpoint year. This quantifies the PRECISION
     of the timing conditional on a break existing; for HBR the existence
     question is addressed separately in analysis 4.

  2. Cross-method comparison.
     Dominant single breakpoint from PELT, binary segmentation (Binseg),
     bottom-up (BottomUp), dynamic programming (Dynp) and a window-based scan,
     to test whether the break is robust to algorithm choice.

  3. Expanded penalty sensitivity.
     PELT breakpoints across a wide penalty grid (0.01-10) for both sites.

  4. Hubbard Brook statistical power.
     Quantifies the limited power given only 4 post-2011 observations (2 if the
     final two years are truncated): two-sample effect size, retrospective
     power, minimum detectable effect, and leave-one-out / min-segment-size
     robustness of the 2011 break.

Usage:
    cd Analysis
    python breakpoint_robustness.py

Requires the same data files as master_analysis.py (see ../Data/README.md).
Output: BREAKPOINT_ROBUSTNESS_RESULTS.txt in output/

Author:  Victor Edmonds
Contact: victoredmonds@gmail.com
"""

import os
import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats
from scipy.stats import nct, t as tdist

warnings.filterwarnings('ignore')

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
SEED = 42

_lines = []
def emit(s=""):
    print(s)
    _lines.append(s)

# --------------------------------------------------------------------------
# Data prep (mirrors master_analysis.py Block 7)
# --------------------------------------------------------------------------
def load_bes_annual_median():
    fp = os.path.join(DATA_DIR, "BES_trace-gas-collection_1998_2025.csv")
    d = pd.read_csv(fp)
    d['Date'] = pd.to_datetime(d['Date'])
    d['Year'] = d['Year'].astype(int)
    d['CH4_flux'] = d['CH4_flux'].replace(MISSING_VALUES, np.nan)
    d = d[d['Site'].isin(FOREST_SITES) & d['CH4_flux'].notna()].copy()
    d = d[~d['Site'].isin(HOTSPOT_SITES)].copy()

    def _trim(g):
        mu, sd = g['CH4_flux'].mean(), g['CH4_flux'].std()
        if sd == 0 or np.isnan(sd):
            return g
        return g[np.abs(g['CH4_flux'] - mu) <= OUTLIER_SD_THRESHOLD * sd]
    d = d.groupby(['Site', 'Year'], group_keys=False).apply(_trim)
    s = d.groupby('Year')['CH4_flux'].median()
    return s.values.astype(float), s.index.astype(int).tolist()

def load_hbr_ref_annual_median():
    fp = os.path.join(DATA_DIR, "knb-lter-hbr.207/knb-lter-hbr.207-CH4_flux_annual.csv")
    d = pd.read_csv(fp)
    if 'Annual CH4flux' in d.columns:
        d = d.rename(columns={'Annual CH4flux': 'Annual_CH4_flux'})
    d['Year'] = d['Year'].astype(int)
    ref = (d[d['Site'] == 'WS6-BB']
           .groupby('Year')['Annual_CH4_flux'].median()
           .reset_index().sort_values('Year'))
    return ref['Annual_CH4_flux'].values.astype(float), ref['Year'].astype(int).tolist()

def dominant_bp(sig, yr, model="l2", min_size=2):
    """Single best breakpoint (Dynp, n_bkps=1). Returns the year (last year of
    the first segment), matching the convention used in master_analysis.py."""
    algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(sig)
    b = [x for x in algo.predict(n_bkps=1) if x < len(sig)]
    return (yr[b[0] - 1], b[0]) if b else (None, None)

# --------------------------------------------------------------------------
# 1. Bootstrap CIs
# --------------------------------------------------------------------------
def bootstrap_ci(sig, yr, label, B=2000, L=3):
    n = len(sig)
    y0, k0 = dominant_bp(sig, yr)
    seg1, seg2 = sig[:k0].mean(), sig[k0:].mean()
    fit = np.concatenate([np.full(k0, seg1), np.full(n - k0, seg2)])
    resid = sig - fit
    rng = np.random.default_rng(SEED)
    nblocks = int(np.ceil(n / L))
    years = []
    for _ in range(B):
        starts = rng.integers(0, n - L + 1, size=nblocks)
        rb = np.concatenate([resid[s:s + L] for s in starts])[:n]
        yb, _ = dominant_bp(fit + rb, yr)
        if yb is not None:
            years.append(yb)
    years = np.array(years)
    lo, hi = np.percentile(years, [2.5, 97.5])
    vals, counts = np.unique(years, return_counts=True)
    top = sorted(zip(vals.tolist(), counts.tolist()), key=lambda kv: -kv[1])[:5]
    emit(f"  [{label}] observed dominant breakpoint = {y0}")
    emit(f"    residual moving-block bootstrap (L={L}, B={len(years)}): "
         f"median={int(np.median(years))}, mode={int(top[0][0])}, "
         f"95% CI = [{int(lo)}, {int(hi)}]")
    emit("    P(year): " + ", ".join(f"{int(v)}={c/len(years)*100:.0f}%" for v, c in top))
    return y0, (int(lo), int(hi))

# --------------------------------------------------------------------------
# 2. Cross-method
# --------------------------------------------------------------------------
def cross_method(sig, yr, label):
    emit(f"  [{label}]")
    res = {}
    b = [x for x in rpt.Pelt(model="l2").fit(sig).predict(pen=0.1) if x < len(sig)]
    res['PELT (pen=0.1)'] = [yr[i - 1] for i in b]
    for name, algo in [("Binseg (n=1)", rpt.Binseg(model="l2", min_size=2, jump=1)),
                       ("BottomUp (n=1)", rpt.BottomUp(model="l2", min_size=2, jump=1)),
                       ("Dynp (n=1)", rpt.Dynp(model="l2", min_size=2, jump=1))]:
        b = [x for x in algo.fit(sig).predict(n_bkps=1) if x < len(sig)]
        res[name] = [yr[i - 1] for i in b]
    try:
        b = [x for x in rpt.Window(width=4, model="l2", jump=1).fit(sig).predict(n_bkps=1) if x < len(sig)]
        res['Window (w=4, n=1)'] = [yr[i - 1] for i in b]
    except Exception as e:
        res['Window (w=4, n=1)'] = f"err: {e}"
    for k, v in res.items():
        emit(f"    {k:20s}: {v}")
    return res

# --------------------------------------------------------------------------
# 3. Penalty sensitivity
# --------------------------------------------------------------------------
def penalty_sweep(sig, yr, label):
    pens = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7, 10]
    algo = rpt.Pelt(model="l2").fit(sig)
    emit(f"  [{label}]  penalty : breakpoints (years)")
    for pen in pens:
        b = [x for x in algo.predict(pen=pen) if x < len(sig)]
        emit(f"    {pen:>5}: {[yr[i - 1] for i in b] if b else 'none'}")

# --------------------------------------------------------------------------
# 4. HBR power
# --------------------------------------------------------------------------
def power_tt(n1, n2, d, alpha=0.05):
    df = n1 + n2 - 2
    ncp = abs(d) * np.sqrt(n1 * n2 / (n1 + n2))
    crit = tdist.ppf(1 - alpha / 2, df)
    return float(1 - nct.cdf(crit, df, ncp) + nct.cdf(-crit, df, ncp))

def mde(n1, n2, target=0.8):
    for dd in np.linspace(0.05, 15, 3000):
        if power_tt(n1, n2, dd) >= target:
            return dd
    return float('nan')

def hbr_power(sig, yr):
    yrs = np.array(yr)
    pre = sig[yrs <= 2011]
    post = sig[yrs > 2011]
    post2 = sig[(yrs > 2011) & (yrs <= 2013)]
    sp = np.sqrt(((len(pre) - 1) * pre.std(ddof=1) ** 2 +
                  (len(post) - 1) * post.std(ddof=1) ** 2) / (len(pre) + len(post) - 2))
    d = (post.mean() - pre.mean()) / sp
    t, p = stats.ttest_ind(pre, post, equal_var=True)
    emit(f"    pre (<=2011):  n={len(pre)}, mean={pre.mean():.3f}, sd={pre.std(ddof=1):.3f}")
    emit(f"    post (2012-2015): n={len(post)}, mean={post.mean():.3f}, sd={post.std(ddof=1):.3f}")
    emit(f"    truncated post (2012-2013): n={len(post2)}, mean={post2.mean():.3f}")
    emit(f"    Cohen's d (pre vs post) = {d:.2f}; two-sample t = {t:.2f}, p = {p:.4f}")
    emit(f"    retrospective power (a=0.05): n_post=4 -> {power_tt(len(pre),4,d):.2f}; "
         f"n_post=2 -> {power_tt(len(pre),2,d):.2f}")
    emit(f"    minimum detectable d at 80% power: n_post=4 -> {mde(len(pre),4):.2f}; "
         f"n_post=2 -> {mde(len(pre),2):.2f}")
    loo = []
    for dy in [2012, 2013, 2014, 2015]:
        keep = yrs != dy
        yb, _ = dominant_bp(sig[keep], [int(x) for x in yrs[keep]])
        loo.append(f"drop {dy}->{yb}")
    emit("    leave-one-out (post-break year): " + ", ".join(loo))
    ms = []
    for m in [2, 3, 4, 5]:
        yb, _ = dominant_bp(sig, yr, min_size=m)
        ms.append(f"min_size={m}->{yb}")
    emit("    min-segment-size robustness: " + ", ".join(ms))

# --------------------------------------------------------------------------
def main():
    bes_sig, bes_yr = load_bes_annual_median()
    hbr_sig, hbr_yr = load_hbr_ref_annual_median()

    emit("=" * 70)
    emit("BREAKPOINT ROBUSTNESS CHECKS")
    emit("Series: BES annual median (l2) and HBR WS6-BB annual median (l2),")
    emit("identical to master_analysis.py Block 7 / Figure 4 / Table 4.")
    emit("=" * 70)

    emit("\n1. BLOCK-BOOTSTRAP CONFIDENCE INTERVALS ON BREAKPOINT TIMING")
    emit("-" * 60)
    bootstrap_ci(bes_sig, bes_yr, "BES (n_years=%d)" % len(bes_yr), L=3)
    bootstrap_ci(hbr_sig, hbr_yr, "HBR WS6-BB (n_years=%d)" % len(hbr_yr), L=2)
    emit("\n  Note: this bootstrap quantifies timing precision CONDITIONAL on a")
    emit("  break existing; for HBR the binding question (does a distinct post-2011")
    emit("  regime exist at all) is addressed by the power analysis in section 4.")

    emit("\n2. CROSS-METHOD COMPARISON (dominant single breakpoint)")
    emit("-" * 60)
    mb = cross_method(bes_sig, bes_yr, "BES")
    mh = cross_method(hbr_sig, hbr_yr, "HBR WS6-BB")
    emit("  BES 2002 recovered by PELT/Binseg/BottomUp/Dynp; window-based -> 1999.")
    emit("  HBR 2011 recovered by all five methods.")

    emit("\n3. EXPANDED PENALTY SENSITIVITY (PELT)")
    emit("-" * 60)
    penalty_sweep(bes_sig, bes_yr, "BES")
    penalty_sweep(hbr_sig, hbr_yr, "HBR WS6-BB")

    emit("\n4. HUBBARD BROOK STATISTICAL POWER (4 post-2011 obs; 2 if truncated)")
    emit("-" * 60)
    hbr_power(hbr_sig, hbr_yr)
    emit("\n  Reading: the observed shift is large (d~3.3) and statistically clear,")
    emit("  so the limitation is NOT that the break is a low-power false positive.")
    emit("  Rather, a 4-point (2-point) terminal regime cannot have its level or")
    emit("  stability characterized, and only large shifts (d>=1.8) are detectable")
    emit("  at all. This supports the manuscript's 'putative shift' framing.")

    emit("\n" + "=" * 70)
    out = os.path.join(OUTPUT_DIR, "BREAKPOINT_ROBUSTNESS_RESULTS.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nResults written to {out}")

if __name__ == "__main__":
    main()
