"""Write reports/reproduction_manifest.md: what produces each table and figure of the
IJIES article, and the exact split, fold, date and seed information behind them."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import *
from meta_gate import FOLDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = np.load(os.path.join(OUT, "meta_dataset.npz"), allow_pickle=True)
st, j, T, ok, sts = Z["st"], Z["j"], Z["T"], Z["ok"], [str(s) for s in Z["stations"]]
split = np.where(T < VA0, 0, np.where(T < TE0, 1, 2))


def ym(t):
    return f"{1926 + t // 12}-{t % 12 + 1:02d}"


L = []
w = L.append
w("# Reproduction manifest")
w("")
w("Release `ijies-r1`. Every number in the article comes from the files named here.")
w("Run `python src/run_all.py` to rebuild all of them from the raw records.")
w("")
w("## 1. Splits")
w("")
w("| Split | Target months | Rows | Observed target | Unobserved target |")
w("|---|---|---|---|---|")
for s_, nm in [(0, "Meta-training"), (1, "Validation"), (2, "Test")]:
    m = split == s_
    tt = T[m]
    w(f"| {nm} | {ym(tt.min())} to {ym(tt.max())} | {m.sum():,} | {(m & ok).sum():,} | {(m & ~ok).sum():,} |")
w(f"| Total | {ym(T.min())} to {ym(T.max())} | {len(T):,} | {ok.sum():,} | {(~ok).sum():,} |")
w("")
w("Rows whose target month fails the observation mask are never training, validation or")
w("test targets. They stay in the dataset because their rainfall enters the error history")
w("of later origins. Split boundaries live in `src/common.py` as `VA0` and `TE0`.")
w("")
w("## 2. Horizon-specific test counts")
w("")
w("| Horizon | Observed test targets, 15 stations | Evaluated, 13 stations |")
w("|---|---|---|")
excl = {"Coffs_Harbour", "Melbourne"}
keep = np.array([sts[i] not in excl for i in st])
for jj, h in enumerate(HORIZONS):
    m = (split == 2) & ok & (j == jj)
    w(f"| h = {h} | {m.sum():,} | {(m & keep).sum():,} |")
m = (split == 2) & ok
w(f"| All | {m.sum():,} | {(m & keep).sum():,} |")
w("")
w("## 3. Observation mask and per-station test months")
w("")
w("Source: `data/processed/monthly_15.json`, field `ok` per station; a month is a valid")
w("target when at least 90 per cent of its days carry a SILO observation code.")
w("")
w("| Station | Observed test months at h = 1 | Evaluated |")
w("|---|---|---|")
for i, s_ in enumerate(sts):
    n = int(((split == 2) & ok & (j == 0) & (st == i)).sum())
    w(f"| {s_.replace('_', ' ')} | {n} | {'no' if s_ in excl else 'yes'} |")
w("")
w("## 4. Transfer folds")
w("")
w("Defined in `src/meta_gate.py` as `FOLDS`. For each fold the gate trains on the other")
w("twelve stations; the pooled NNLS anchor also uses training stations only.")
w("")
w("| Fold | Held-out stations | Prediction file |")
w("|---|---|---|")
for f, g in enumerate(FOLDS):
    w(f"| {f} | {', '.join(x.replace('_', ' ') for x in g)} | `results/pred_transfer_{f}.npz` |")
w("")
w("## 5. Forecast origins, target dates and scaling fits")
w("")
w("`results/meta_dataset.npz` carries one row per station, horizon and target month:")
w("`T` is the target month index (0 = Jan 1926, so " + ym(0) + "), `j` indexes the horizon,")
w("`st` indexes the station, and the forecast origin is `T - h + 1`. `SC` is the station")
w("scale, the standard deviation of observed monthly rainfall before " + ym(VA0) + ".")
w("Base-model scaling and climate-index standardisation are refitted at every refit from")
w("months preceding it only; see `src/base_models_rolling.py`.")
w("")
w("Refit months: " + ", ".join(ym(s_) for s_ in range(R0, N, STEP)) + ".")
w("")
w("## 6. Per-seed gate predictions")
w("")
w("Gate seeds are 1, 2 and 3 (`SEEDS` in `src/meta_gate.py`); reported values average them.")
w("Each file stores predictions with the seed on the first axis.")
w("")
w("| File | Contents |")
w("|---|---|")
for f in sorted(glob.glob(os.path.join(OUT, "pred_local_*.npz"))):
    n = os.path.basename(f)
    w(f"| `results/{n}` | L-G3 and L-G4 for {n[11:-4].replace('_', ' ')}, shape (seed, sample) |")
for n in ["pred_global_GG3.npz", "pred_global_GG4.npz"]:
    if os.path.exists(os.path.join(OUT, n)):
        w(f"| `results/{n}` | {n[12:-4]} over all stations, shape (seed, sample) |")
for f in range(len(FOLDS)):
    w(f"| `results/pred_transfer_{f}.npz` | GG4-T on the held-out stations of fold {f} |")
w("")
w("## 7. Tables and figures of the article")
w("")
w("Numbering follows the revised (R1) manuscript. Files keep the names of the submitted")
w("version, so the stored `table2_*` file is R1 Table 3, and so on.")
w("")
w("| R1 item | Produced by | Stored as |")
w("|---|---|---|")
rows = [
 ("Table 1, study stations", "src/manuscript_tables_figures.py", "reports/manuscript/table1_stations.csv"),
 ("Table 2, split flow", "src/make_manifest.py", "this manifest, section 1"),
 ("Table 3, mean NSE, RMSE, MAE", "src/evaluate.py, src/evaluate_errors.py", "reports/manuscript/table2_nse_mean.csv, table4_rmse_mae.csv"),
 ("Table 4, panel DM, squared error", "src/evaluate.py", "reports/manuscript/table3_dm_panel.csv"),
 ("Table 5, panel DM, absolute error", "src/evaluate_errors.py", "reports/manuscript/table5_dm_absolute_error.csv"),
 ("Table 6, per-station NSE", "src/evaluate.py", "reports/manuscript/table6_nse_stations.csv"),
 ("Figure 1, workflow", "drawn separately", "reports/manuscript/figure1_framework.png"),
 ("Figure 2, skill by horizon", "src/manuscript_tables_figures.py", "reports/manuscript/figure2_skill.png"),
 ("Figure 3, per-station heatmap", "src/manuscript_tables_figures.py", "reports/manuscript/figure3_heatmap.png"),
 ("Figure 4, learned weights", "src/manuscript_tables_figures.py", "reports/manuscript/figure4_weights.png"),
]
for a, b, c in rows:
    w(f"| {a} | `{b}` | `{c}` |")
w("")
w("## 8. Panel loss series behind each Diebold-Mariano test")
w("")
w("`src/evaluate.py` builds, for every pair of methods and every horizon, the series of")
w("cross-station mean scaled squared-error differences indexed by target month, then")
w("applies `dm_test` from `src/common.py` with the long-run variance truncated at h - 1")
w("and the Harvey-Leybourne-Newbold correction. `src/evaluate_errors.py` repeats this on")
w("scaled absolute errors. Both write the per-horizon statistic, raw p-value and")
w("Holm-adjusted p-value to `results/evaluation_results.json` and")
w("`results/error_metrics.json` under the key `panel`.")
w("")

out = os.path.join(REPORTS, "reproduction_manifest.md")
os.makedirs(REPORTS, exist_ok=True)
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", out, len(L), "lines")
