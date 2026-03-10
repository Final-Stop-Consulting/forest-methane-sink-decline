# Data Manifest

This document describes every dataset used by `Analysis/master_analysis.py`, where to get it, and where to put it. All data are publicly available. No accounts or special access required.

The script expects all files to live under `Data/` relative to the project root. The project structure is:

```
Forest Biofilter/
├── Analysis/
│   ├── master_analysis.py
│   └── output/
├── Data/                          ← everything below goes here
│   ├── BES_trace-gas-collection_1998_2025.csv
│   ├── NADP/
│   ├── PRISM/
│   ├── knb-lter-bes.3300.110/
│   ├── knb-lter-bes.3400/
│   ├── knb-lter-bes.428.292/
│   ├── knb-lter-bes.584/
│   ├── knb-lter-hbr.207/
│   └── knb-lter-hfr.60.19/
```

---

## 1. BES Trace Gas (CH₄ flux)

**What:** 27 years of static chamber CH₄ flux measurements from the Baltimore Ecosystem Study, 1998–2025. This is the primary dataset.

**Source:** BES LTER Data Archive
- Package: `knb-lter-bes.585.654`
- Portal: https://portal.lternet.edu/ → search "knb-lter-bes.585"

**File:** `BES_trace-gas-collection_1998_2025.csv`
**Place at:** `Data/BES_trace-gas-collection_1998_2025.csv`

A copy also exists at `Data/knb-lter-bes.585.654/BES_trace-gas-collection_1998_2025.csv`. The script reads from the root-level copy.

**Key columns used:** `Site`, `CollectionDate`, `CH4_flux`

---

## 2. Hubbard Brook CH₄ Flux

**What:** CH₄ flux from Hubbard Brook Experimental Forest, 2002–2015. Annual and monthly data from Watershed 1 (Ca-treated) and Watershed 6 / Bear Brook (reference).

**Source:** Hubbard Brook LTER Data Archive
- Package: `knb-lter-hbr.207`
- Portal: https://portal.lternet.edu/ → search "knb-lter-hbr.207"

**Files:**
- `knb-lter-hbr.207-CH4_flux_annual.csv` — annual means by watershed
- `knb-lter-hbr.207-CH4_flux_monthly.csv` — monthly means with replicates

**Place at:** `Data/knb-lter-hbr.207/`

**Used in:** Block 5 (calcium experiment), Block 7 (breakpoint detection)

---

## 3. PRISM Climate Data

**What:** Monthly precipitation (ppt) and mean temperature (tmean) at 4 km resolution for each study site, 1998–2025.

**Source:** PRISM Climate Group, Oregon State University
- https://prism.oregonstate.edu/explorer/
- Select "Time Series Values" for a point location, monthly, ppt + tmean

**Files and coordinates:**

| File | Site | Lat | Lon |
|------|------|-----|-----|
| `BES-PRISM_ppt_tmean_stable_4km_199801_202506_39.3400_-76.6200.csv` | BES urban sites (HD, LEA, MCD, GB, GLY, UMBC) | 39.34°N | 76.62°W |
| `OregonRidge-PRISM_ppt_tmean_stable_4km_199801_202506_39.4970_-76.6890.csv` | BES rural sites (ORM, ORU, ORLR, ORUR, CAH) | 39.497°N | 76.689°W |
| `HubbardBrook-PRISM_ppt_tmean_stable_4km_199801_202506_43.9440_-71.7500.csv` | Hubbard Brook | 43.944°N | 71.750°W |

**Place at:** `Data/PRISM/`

**Note:** PRISM CSVs have 10 header rows of metadata. The script skips them with `skiprows=10`. Two additional PRISM files exist in the folder (Harvard Forest, Howland Forest) but are not used in the current analysis.

---

## 4. NADP Wet Deposition

**What:** Monthly wet deposition of sulfate and inorganic nitrogen from the National Atmospheric Deposition Program (NADP) National Trends Network.

**Source:** https://nadp.slh.wisc.edu/
- NTN Data → select station → download monthly data in kg/ha

**Files:**

| File | Station | What |
|------|---------|------|
| `NTN-nh02-m-s-kg.csv` | NH02 (Hubbard Brook) | SO₄ deposition, 1978–2023 |
| `NTN-md99-m-s-kg.csv` | MD99 (Beltsville, MD) | Inorganic N deposition, 1999–2023 |

**Place at:** `Data/NADP/`

**Used in:** Block 8 (deposition overlay with flux time series)

---

## 5. BES Soil Moisture (VWC)

**What:** Hourly volumetric water content and soil temperature from BES sensor networks, 2011–2020. 691,965 readings across 9 sensors at 5 sites.

**Source:** BES LTER Data Archive
- Package: `knb-lter-bes.3400`

**File:** `BES_TempVWC_2011-2020.csv`
**Place at:** `Data/knb-lter-bes.3400/`

**Used in:** Block 11 (direct VWC vs. CH₄ flux test — the mechanistically relevant variable)

---

## 6. BES Lysimeter (Soil Nitrate)

**What:** Lysimeter NO₃⁻ concentrations from three BES forested sites (Hillsdale, Leakin Park, Oregon Ridge), 1999–2025. 20,479 measurements.

**Source:** BES LTER Data Archive
- Package: `knb-lter-bes.428.292`

**File:** `BES_lysimeter_data_1999-2025_for_EDI.csv`
**Place at:** `Data/knb-lter-bes.428.292/`

**Used in:** Block 9 (soil nitrogen trends)

---

## 7. BES Vegetation (Seedlings)

**What:** Seedling cover data from three BES forest census years (1998, 2003, 2015).

**Source:** BES LTER Data Archive
- Package: `knb-lter-bes.3300.110`

**File used:** `vegetation_BESLTER_1998_2003_2015_Veg_Data_9_Tree_Seedling_Cover.csv`
**Place at:** `Data/knb-lter-bes.3300.110/`

**Note:** The download includes 13 CSV files covering trees, saplings, herbs, shrubs, vines, etc. Only the seedling cover file (Veg_Data_9) is used. Keep the full package for completeness.

**Used in:** Block 10 (vegetation community context)

---

## 8. BES Soil Properties

**What:** Physical, chemical, and biological properties of forest soils at 0–10 cm depth for four BES sites (HD, LEA, ORM, ORU). Includes C:N, microbial biomass C, net N mineralization, net nitrification.

**Source:** BES LTER Data Archive
- Package: `knb-lter-bes.584`

**File:** `Physical_chemical_and_biological_properties_of_forest_and_home_lawn_soils.csv`
**Place at:** `Data/knb-lter-bes.584/`

**Used in:** Block 13 (soil properties characterization)

---

## 9. Harvard Forest Atmospheric CH₄

**What:** Ambient atmospheric CH₄ concentration measurements from Harvard Forest, 1991–1994. Baseline context only (not soil flux).

**Source:** Harvard Forest LTER Data Archive
- Package: `knb-lter-hfr.60.19`
- https://harvardforest.fas.harvard.edu/data-archives

**File used:** `hf060-02-methane.csv`
**Place at:** `Data/knb-lter-hfr.60.19/`

**Used in:** Block 12 (independent atmospheric CH₄ trend context)

---

## Running the analysis

```bash
cd Analysis/
pip install -r requirements.txt
python master_analysis.py
```

Output goes to `Analysis/output/`: 14 figures (PNG + SVG) and `SUMMARY.txt`.

Runtime is ~2 minutes on a standard laptop. No GPU required.
