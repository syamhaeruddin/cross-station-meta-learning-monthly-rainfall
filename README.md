# Cross-Station Meta-Learning for Dynamic Combination of Monthly Rainfall Forecasts

Code, data, and results for the manuscript *"Cross-Station Meta-Learning for Dynamic Combination of Monthly Rainfall Forecasts"* (Haeruddin, Purwanto, Muljono, Edi Noersasongko).

Dynamic weighting of monthly rainfall forecast combinations rarely beats a simple average or fixed weights. One reason is that the meta-sample available at a single station is too small to train a meta-learner. This study trains one **global meta-learner gate** jointly on 15 SILO stations in Australia (1926–2025). The gate assigns convex weights to an 11-member pool: nine statistical, machine-learning, deep-learning, and causal-decomposition hybrid models, each retrained every five years, plus two rolling climatologies. The global gate is compared with a local gate of identical architecture, NNLS stacking, the simple average, and a 30-year rolling climatology, and is then tested for transfer to stations that were held out from training.

The code, notebook, and column headers are written in Indonesian (the language of the dissertation); this README and `data/README.md` are in English. A glossary of the Indonesian identifiers used in the code is given at the end.

## Main results (test period 2015–2025, 13 stations)

| Horizon | Klim30 | Mean | NNLS | Local gate L-G3 | **Global gate GG3** | Transfer gate GG4-T |
|---|---|---|---|---|---|---|
| 1 month | 0.300 | 0.339 | 0.330 | 0.326 | **0.340** | 0.338 |
| 3 months | 0.300 | 0.326 | 0.327 | 0.329 | **0.340** | 0.332 |
| 6 months | 0.300 | 0.310 | 0.322 | 0.323 | **0.333** | 0.329 |
| 12 months | 0.300 | 0.308 | 0.321 | 0.321 | **0.329** | 0.324 |

Entries are the NSE averaged across stations.

- **GG3 vs L-G3:** GG3 is more accurate in 39 of 52 station–horizon combinations. The panel Diebold–Mariano test is significant at horizons 1, 3, and 6 months (Holm-adjusted *p* = 0.008, 0.009, 0.033).
- **GG3 vs static benchmarks:** GG3 beats Klim30 at horizons 1 and 3 months (*p* = 0.006) and NNLS at horizon 3 months (*p* = 0.005). These comparisons are exploratory.
- **Transfer:** a gate trained without the test station (GG4-T) performs on par with a gate that saw that station.
- **RMSE and MAE:** GG3 also has the lowest mean RMSE (87.1–88.1 mm) and MAE (56.4–57.4 mm) at all four horizons. With absolute errors, the advantage over L-G3 persists at horizon 1 month (Holm *p* = 0.005) and the advantage over NNLS strengthens (*p* = 0.003 and 0.008 at horizons 1 and 3), but the advantage over Klim30 is no longer significant: GG3's improvement is concentrated in wet months.
- **Effect size:** small, with a median MSE reduction of 1.4–3.3%.

## Repository layout

```
├── notebooks/
│   └── Eksperimen_Gate_Global_Lintas_Stasiun.ipynb   # full pipeline: data -> models -> gate -> evaluation -> tables & figures
├── src/
│   ├── common.py              # constants, file locations, NSE, DM-HLN test, Holm correction
│   ├── siapkan_data.py        # stage 1: daily rainfall -> monthly totals + observation mask
│   ├── indeks_iklim.py        # stage 2: Niño3.4 (ERSST v5), DMI (HadISST), SOI (CRU)
│   ├── base_roll.py           # stage 3: 9 base models, rolling retraining every 5 years
│   ├── meta_global.py         # stage 4: meta-dataset, local, global, and transfer gates
│   ├── eval_global.py         # stage 5: NSE, per-station DM tests (Holm), panel DM tests, diagnostics
│   ├── eval_galat.py          # stage 5b: RMSE, MAE, MAE skill, panel DM tests with squared and absolute errors
│   ├── laporan_eksperimen.py  # stage 6a: tables and figures of the full experiment report
│   ├── tabel_gambar_naskah.py # stage 6b: Tables 1-6 and Figures 1-4 of the manuscript
│   └── run_all.py             # runs stages 3-5 in parallel from the command line
├── data/
│   ├── mentah/                # raw SILO data and climate indices (see data/README.md)
│   └── olahan/                # monthly rainfall, observation mask, climate indices
├── hasil/                     # base-model outputs, gate predictions, hasil_evaluasi.json, hasil_galat.json
├── laporan/
│   ├── naskah/                # manuscript Tables 1-6 (CSV) and Figures 1-4 (PNG)
│   └── eksperimen/            # all experiment tables and figures, incl. tabel_gate_global.xlsx
├── requirements.txt
└── README.md
```

## How to run

Python 3.13 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/Eksperimen_Gate_Global_Lintas_Stasiun.ipynb
```

The notebook has two switches in its first code cell:

| Mode | `LATIH_ULANG_MODEL_DASAR` (retrain base models) | `LATIH_ULANG_GATE` (retrain gates) | Time (16-core CPU) |
|---|---|---|---|
| Reproduction (default) | `False` | `False` | ≈ 3 minutes |
| Full pipeline | `True` | `True` | ≈ 25 minutes with 15 parallel processes; ≈ 4–5 hours with 1 process |

Reproduction mode rebuilds the derived data from the raw data, assembles the meta-dataset from the base-model outputs in `hasil/`, runs the evaluation (NSE, RMSE, MAE, and DM tests), and regenerates every table and figure. The last section of the notebook checks the figures quoted in the manuscript.

The same pipeline can be run from the command line without the notebook:

```bash
python src/siapkan_data.py
python src/indeks_iklim.py
python src/run_all.py all          # base models, gates, evaluation (existing files are skipped)
python src/tabel_gambar_naskah.py
python src/laporan_eksperimen.py
```

The number of parallel processes is set with the environment variable `WORKERS`, and the gate seeds with `SEEDS` (default `1,2,3`).

## Protocol in brief

- **Observation mask:** a month is a training/evaluation target only if ≥ 90% of its days carry a SILO observation code (0, 13, 15, 23). Interpolated months may still serve as inputs.
- **Index release latency:** Niño3.4 and DMI are used up to month *o* − 2, SOI up to *o* − 1.
- **Base models:** retrained every 5 years from January 1961 (13 refits). Every forecast from 1961 onwards is out of sample. VMD is rerun at every origin on past data only, and the à trous Haar wavelet is causal.
- **Meta periods:** meta-training 1962–2009, validation (early stopping) 2010–2014, test 2015–2025 (sub-periods 2015–2020 and 2021–2025).
- **Statistical tests:**
  - DM with the Harvey–Leybourne–Newbold correction per station, Holm-adjusted over 52 combinations.
  - Panel DM: scaled squared-error differences averaged across stations per month, Holm-adjusted over 4 horizons.
  - Robustness: the same panel DM with scaled absolute-error differences; RMSE, MAE, and MAE skill relative to Klim30 (computed per station, then averaged).
- **Local gates without observed validation months:** Adelaide and Birdsville have no observed target months in 2010–2014. Their local gates use the last 20% of observed training targets for early stopping instead of an empty validation set (`src/meta_global.py`).

## Reproducibility notes

- **Reproduction mode:** the base-model and gate outputs in `hasil/` were produced on the author's Windows machine with the library versions in `requirements.txt`. From those outputs, reproduction mode regenerates identical derived data, the same `hasil_evaluasi.json` (numerical differences < 10⁻¹² between Windows and Linux), and pixel-identical manuscript figures.
- **Retraining on another Linux machine** with the same library versions (tested for one station's base models and for all three gate types):
  - derived data and the meta-dataset are identical;
  - Woomera base models: SVR and the four XGBoost models are identical, LSTM and GRU differ by < 10⁻⁴ mm, but SARIMA differs by 0.6 mm on average (≈ 4% of the forecast value) because statsmodels' optimiser is platform dependent;
  - local and transfer gates are reproduced to float32 precision (differences < 10⁻⁶);
  - the global gate GG3 differs because its training is sensitive to small numerical differences between machines, so *p*-values near the 0.05 threshold may shift;
  - in that test, GG3 vs L-G3 remained significant at horizons 1 and 3 months but not at 6 months, and GG3's advantage over NNLS in 2021–2025 remained significant at horizons 1 and 3 months but not at 6 months;
  - increasing the number of gate seeds (e.g. `SEEDS=1,2,3,4,5,6,7,8,9,10`) reduces this sensitivity.
- **Meta-dataset** (`hasil/meta_dataset.npz`, ≈ 58 MB) is not stored in the repository; the notebook rebuilds it in seconds.
- **Status of additional analyses:** the comparisons of GG3 with the mean, NNLS, and Klim30, and the RMSE, MAE, and absolute-error tests, were added after the initial results were seen and are therefore treated as exploratory in the manuscript.

## Glossary of identifiers used in the code

| Indonesian | English |
|---|---|
| `mentah` / `olahan` | raw / derived (data) |
| `hasil` | results |
| `laporan`, `naskah` | report, manuscript |
| `model_dasar` | base model |
| `gate lokal` / `global` / `transfer` | local / global / transfer gate |
| `Rata2`, `Rata-rata` | simple mean of the base models |
| `Klim30`, `Klim-30th` | 30-year rolling climatology |
| `Klim-ekspansif` | expanding-window climatology |
| `latih` / `validasi` / `uji` | train / validation / test |
| `lipatan` | fold |
| `galat` | error |
| `menang` | wins (number of station–horizon combinations won) |
| `bobot` | weight |
| `stasiun`, `horizon`, `bulan` | station, horizon, month |

## Data sources

Rainfall: Queensland Government SILO (CC BY 4.0). Climate indices: NOAA NCEI (ERSST v5), NOAA CPC, NOAA PSL (DMI from HadISST), and the Climatic Research Unit (SOI). Details, checksums, and attribution are in [`data/README.md`](data/README.md).

## Contact

Haeruddin, Faculty of Computer Science, Universitas Dian Nuswantoro, Semarang, and Universitas Internasional Batam — haeruddin@uib.ac.id
