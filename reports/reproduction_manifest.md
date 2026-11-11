# Reproduction manifest

Release `ijies-r2`. Every number in the article comes from the files named here.
Run `python src/run_all.py` to rebuild all of them from the raw records.

## 1. Splits

| Split | Target months | Rows | Observed target | Unobserved target |
|---|---|---|---|---|
| Meta-training | 1962-02 to 2009-12 | 33,960 | 31,160 | 2,800 |
| Validation | 2010-01 to 2014-12 | 3,600 | 3,092 | 508 |
| Test | 2015-01 to 2025-12 | 7,920 | 6,096 | 1,824 |
| Total | 1962-02 to 2025-12 | 45,480 | 40,348 | 5,132 |

Rows whose target month fails the observation mask are never training, validation or
test targets. They stay in the dataset because their rainfall enters the error history
of later origins. Split boundaries live in `src/common.py` as `VA0` and `TE0`.

## 2. Horizon-specific test counts

| Horizon | Observed test targets, 15 stations | Evaluated, 13 stations |
|---|---|---|
| h = 1 | 1,524 | 1,517 |
| h = 3 | 1,524 | 1,517 |
| h = 6 | 1,524 | 1,517 |
| h = 12 | 1,524 | 1,517 |
| All | 6,096 | 6,068 |

## 3. Observation mask and per-station test months

Source: `data/processed/monthly_15.json`, field `ok` per station; a month is a valid
target when at least 90 per cent of its days carry a SILO observation code.

| Station | Observed test months at h = 1 | Evaluated |
|---|---|---|
| Darwin | 132 | yes |
| Katherine | 125 | yes |
| Broome | 131 | yes |
| Cairns | 127 | yes |
| Innisfail | 66 | yes |
| Tully | 132 | yes |
| Brisbane | 130 | yes |
| Coffs Harbour | 7 | no |
| Alice Springs | 129 | yes |
| Woomera | 128 | yes |
| Birdsville | 51 | yes |
| Perth | 132 | yes |
| Adelaide | 102 | yes |
| Melbourne | 0 | no |
| Hobart | 132 | yes |

## 4. Transfer folds

Defined in `src/meta_gate.py` as `FOLDS`. For each fold the gate trains on the other
twelve stations; the pooled NNLS anchor also uses training stations only.

| Fold | Held-out stations | Prediction file |
|---|---|---|
| 0 | Darwin, Brisbane, Woomera | `results/pred_transfer_0.npz` |
| 1 | Perth, Cairns, Birdsville | `results/pred_transfer_1.npz` |
| 2 | Tully, Alice Springs, Coffs Harbour | `results/pred_transfer_2.npz` |
| 3 | Hobart, Katherine, Adelaide | `results/pred_transfer_3.npz` |
| 4 | Broome, Innisfail, Melbourne | `results/pred_transfer_4.npz` |

## 5. Forecast origins, target dates and scaling fits

`results/meta_dataset.npz` carries one row per station, horizon and target month:
`T` is the target month index (0 = Jan 1926, so 1926-01), `j` indexes the horizon,
`st` indexes the station, and the forecast origin is `T - h + 1`. `SC` is the station
scale, the standard deviation of observed monthly rainfall before 2010-01.
Base-model scaling and climate-index standardisation are refitted at every refit from
months preceding it only; see `src/base_models_rolling.py`.

Refit months: 1961-01, 1966-01, 1971-01, 1976-01, 1981-01, 1986-01, 1991-01, 1996-01, 2001-01, 2006-01, 2011-01, 2016-01, 2021-01.

## 6. Per-seed gate predictions

Gate seeds are 1, 2 and 3 (`SEEDS` in `src/meta_gate.py`); reported values average them.
Each file stores predictions with the seed on the first axis.

| File | Contents |
|---|---|
| `results/pred_local_Adelaide.npz` | L-G3 and L-G4 for Adelaide, shape (seed, sample) |
| `results/pred_local_Alice_Springs.npz` | L-G3 and L-G4 for Alice Springs, shape (seed, sample) |
| `results/pred_local_Birdsville.npz` | L-G3 and L-G4 for Birdsville, shape (seed, sample) |
| `results/pred_local_Brisbane.npz` | L-G3 and L-G4 for Brisbane, shape (seed, sample) |
| `results/pred_local_Broome.npz` | L-G3 and L-G4 for Broome, shape (seed, sample) |
| `results/pred_local_Cairns.npz` | L-G3 and L-G4 for Cairns, shape (seed, sample) |
| `results/pred_local_Coffs_Harbour.npz` | L-G3 and L-G4 for Coffs Harbour, shape (seed, sample) |
| `results/pred_local_Darwin.npz` | L-G3 and L-G4 for Darwin, shape (seed, sample) |
| `results/pred_local_Hobart.npz` | L-G3 and L-G4 for Hobart, shape (seed, sample) |
| `results/pred_local_Innisfail.npz` | L-G3 and L-G4 for Innisfail, shape (seed, sample) |
| `results/pred_local_Katherine.npz` | L-G3 and L-G4 for Katherine, shape (seed, sample) |
| `results/pred_local_Melbourne.npz` | L-G3 and L-G4 for Melbourne, shape (seed, sample) |
| `results/pred_local_Perth.npz` | L-G3 and L-G4 for Perth, shape (seed, sample) |
| `results/pred_local_Tully.npz` | L-G3 and L-G4 for Tully, shape (seed, sample) |
| `results/pred_local_Woomera.npz` | L-G3 and L-G4 for Woomera, shape (seed, sample) |
| `results/pred_global_GG3.npz` | GG3 over all stations, shape (seed, sample) |
| `results/pred_global_GG4.npz` | GG4 over all stations, shape (seed, sample) |
| `results/pred_transfer_0.npz` | GG4-T on the held-out stations of fold 0 |
| `results/pred_transfer_1.npz` | GG4-T on the held-out stations of fold 1 |
| `results/pred_transfer_2.npz` | GG4-T on the held-out stations of fold 2 |
| `results/pred_transfer_3.npz` | GG4-T on the held-out stations of fold 3 |
| `results/pred_transfer_4.npz` | GG4-T on the held-out stations of fold 4 |

## 7. Tables and figures of the article

Numbering follows the revised (R1) manuscript. Files keep the names of the submitted
version, so the stored `table2_*` file is R1 Table 3, and so on.

| R2 item | Produced by | Stored as |
|---|---|---|
| Table 1, study stations | `src/manuscript_tables_figures.py` | `reports/manuscript/table1_stations.csv` |
| Table 2, split flow | `src/make_manifest.py` | `this manifest, section 1` |
| Table 3, average NSE, RMSE, MAE | `src/evaluate.py, src/evaluate_errors.py` | `reports/manuscript/table2_nse_mean.csv, table4_rmse_mae.csv` |
| Table 4, panel DM, squared error | `src/evaluate.py` | `reports/manuscript/table3_dm_panel.csv` |
| Table 5, panel DM, absolute error | `src/evaluate_errors.py` | `reports/manuscript/table5_dm_absolute_error.csv` |
| Table 6, ablations and controls | `src/ablations.py, src/extra_baselines.py` | `results/ablations.json, results/extra_baselines.json` |
| Table 7, learning curves | `src/ablations.py curve, curve2` | `results/ablations.json, results/pred_curve*_*.npz` |
| Table 8, transfer by fold | `src/evaluate.py` | `results/evaluation_results.json, results/pred_transfer_*.npz` |
| Figure 1, workflow | `drawn separately` | `reports/manuscript/figure1_framework.png` |
| Figure 2, skill by horizon | `src/manuscript_tables_figures.py` | `reports/manuscript/figure2_skill.png` |
| Figure 3, learned weights | `src/manuscript_tables_figures.py` | `reports/manuscript/figure4_weights.png` |

Two items of the submitted version are no longer tables or figures of the article.
Section 4.7 reports their content in the text and refers the reader here for the rest:

| Withdrawn from the article | Produced by | Stored as |
|---|---|---|
| per-station NSE, formerly Table 6 | `src/evaluate.py` | `reports/manuscript/table6_nse_stations.csv` |
| per-station MSE reduction heat map | `src/manuscript_tables_figures.py` | `reports/manuscript/figure3_heatmap.png` |

Section 4.8 additionally rests on `results/extra_diagnostics.json` for the bandwidth and
block-bootstrap checks and on `results/pred_abl_*.npz` for the per-seed ablation predictions.

## 8. Panel loss series behind each Diebold-Mariano test

`src/evaluate.py` builds, for every pair of methods and every horizon, the series of
cross-station mean scaled squared-error differences indexed by target month, then
applies `dm_test` from `src/common.py` with the long-run variance truncated at h - 1
and the Harvey-Leybourne-Newbold correction. `src/evaluate_errors.py` repeats this on
scaled absolute errors. Both write the per-horizon statistic, raw p-value and
Holm-adjusted p-value to `results/evaluation_results.json` and
`results/error_metrics.json` under the key `panel`.

