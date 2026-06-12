# Testing the Precipitation-Driven Diffusion Limitation Hypothesis for Declining Methane Uptake in Forest Soils

Code and data manifest for:

**Testing the Precipitation-Driven Diffusion Limitation Hypothesis for Declining Methane Uptake in Forest Soils**

Victor Edmonds. Preprint: [bioRxiv link TBD]

## Summary

Upland forest soils consume 22--38 Tg CH4 yr-1. A 53--89% decline in this sink was documented at two LTER networks (BES, HBR) and attributed to increased precipitation via diffusion limitation (Ni and Groffman, 2018, *PNAS*).

We tested five predictions of the diffusion hypothesis against 27 years of chamber flux data from the Baltimore Ecosystem Study (1998--2025; n = 9,359) and 14 years from Hubbard Brook (2002--2015). Four of five predictions were not supported:

1. Monthly precipitation explains 0.08% of flux variance (R2 = 0.0008)
2. Direct in-situ soil moisture explains 0.55% (R2 = 0.0055)
3. No seasonal moisture--flux structure matches diffusion predictions
4. Urban and rural BES forests diverge under shared precipitation

A fifth test -- the calcium silicate amendment at Hubbard Brook -- produced a null result (Cohen's d = -0.012) consistent with diffusion limitation but also with biological alternatives, constraining the recovery potential of the methanotrophic community.

The convergence of these results suggests that precipitation-driven diffusion limitation does not adequately explain the observed decline, pointing instead toward biological control consistent with nitrogen-mediated degradation of the high-affinity methanotrophic community. (Note: this tests precipitation as the *driver* of the decline; diffusion as a transport mechanism is not itself rejected.)

## Repository structure

```
.
├── Analysis/
│   ├── master_analysis.py          # Complete reproducible analysis (15 blocks)
│   ├── supplemental_robustness.py  # S1-S13 robustness checks: LMM, outlier, quadratic (VWC n=2,415), pre-breakpoint, interaction, nested LMM, AR(1), per-site R², permutation, lysimeter, HBR sensitivity (l2/annual), Precip×Post2002 interaction, random slopes LMM
│   ├── breakpoint_robustness.py    # S14-S17: block-bootstrap breakpoint CIs, cross-method comparison, expanded penalty sweep, HBR power analysis
│   ├── requirements.txt            # Python dependencies
│   └── output/                     # Generated figures (PNG + SVG), SUMMARY.txt, SUPPLEMENTAL_RESULTS.txt, BREAKPOINT_ROBUSTNESS_RESULTS.txt
├── Data/
│   └── README.md                   # Data manifest: sources, download URLs, file placement
└── README.md                       # This file
```

## Reproducing the analysis

### 1. Get the data

All datasets are publicly available. See [`Data/README.md`](Data/README.md) for download links, exact filenames, and where to place each file.

### 2. Install dependencies

```bash
pip install -r Analysis/requirements.txt
```

Requires Python 3.8+. Dependencies: pandas, numpy, scipy, statsmodels, matplotlib, seaborn, ruptures.

### 3. Run

```bash
cd Analysis
python master_analysis.py
python supplemental_robustness.py
python breakpoint_robustness.py
```

Output: 5 manuscript figures + 10 supplementary figures (all PNG + SVG), `SUMMARY.txt`, `SUPPLEMENTAL_RESULTS.txt`, and `BREAKPOINT_ROBUSTNESS_RESULTS.txt` in `Analysis/output/`.

Runtime: ~2 minutes on a standard laptop.

## Data sources

| Dataset | Source | Package ID |
|---------|--------|------------|
| BES CH4 flux (1998--2025) | BES LTER | knb-lter-bes.585.654 |
| HBR CH4 flux (2002--2015) | Hubbard Brook LTER | knb-lter-hbr.207 |
| PRISM climate | Oregon State University | prism.oregonstate.edu |
| NADP wet deposition | NADP NTN | nadp.slh.wisc.edu |
| BES soil moisture (2011--2020) | BES LTER | knb-lter-bes.3400 |
| BES lysimeter NO3 (1999--2025) | BES LTER | knb-lter-bes.428.292 |
| BES vegetation | BES LTER | knb-lter-bes.3300.110 |
| BES soil properties | BES LTER | knb-lter-bes.584 |
| Harvard Forest atm. CH4 | Harvard Forest LTER | knb-lter-hfr.60.19 |

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this code or analysis, please cite the preprint:

> Edmonds, V. (2026). Testing the diffusion limitation hypothesis for declining methane uptake in forest soils. *bioRxiv*. doi: [https://doi.org/10.64898/2026.03.12.711040]

## Data and Code Archive

Archived dataset and analysis code: [https://doi.org/10.5281/zenodo.18944402](https://doi.org/10.5281/zenodo.18944402)
