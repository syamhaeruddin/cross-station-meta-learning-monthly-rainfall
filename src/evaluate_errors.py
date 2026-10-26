# Stage 5b: RMSE, MAE, MAE skill, and panel Diebold-Mariano tests with squared and absolute errors.
# Writes results/error_metrics.json. Run after src/evaluate.py (uses the same meta-dataset and gate predictions).
#   python src/evaluate_errors.py
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from meta_gate import load_ds, station_nnls_prior

D = load_ds(); LP, W = station_nnls_prior(D)
sts = list(D["stations"]); pool = list(D["pool"]); NB = len(pool) - 2
P, y, SC = D["P"], D["y"], D["SC"]

# forecasts of every method (same as src/evaluate.py)
PRED = {"Clim-30yr": P[:, pool.index("Clim-30yr")], "Clim-expanding": P[:, pool.index("Clim-expanding")], "Mean": P[:, :NB].mean(1),
        "NNLS": np.array([P[k] @ W[(D["st"][k], D["j"][k])] for k in range(len(y))])}
best = np.zeros(len(y))
for si in range(len(sts)):
    for j in range(4):
        k = (D["st"] == si) & (D["j"] == j); kp = k & (D["split"] < 2) & D["ok"]
        best[k] = P[k, int(np.argmin(((P[kp] - y[kp, None])**2).mean(0)))]
PRED["Best model"] = best
def put(name, idx, arr):
    full = PRED.setdefault(name, np.full(len(y), np.nan)); full[idx] = arr.mean(0)
for f in glob.glob(os.path.join(OUT, "pred_local_*.npz")):
    Z = np.load(f)
    for nm in ["L-G3", "L-G4"]: put(nm, Z["idx"], Z[nm])
for nm in ["GG3", "GG4"]:
    Z = np.load(os.path.join(OUT, f"pred_global_{nm}.npz")); put(nm, Z["idx"], Z["pred"])
for f in glob.glob(os.path.join(OUT, "pred_transfer_*.npz")):
    Z = np.load(f); put("GG4-transfer", Z["idx"], Z["pred"])
METHODS = ["Clim-30yr", "Clim-expanding", "Mean", "NNLS", "Best model", "L-G3", "L-G4", "GG3", "GG4", "GG4-transfer"]
TESTSTS = [s for s in sts if s not in ("Coffs_Harbour", "Melbourne")]
PER = {"2015-2025": (TE0, N), "2015-2020": (TE0, 1140), "2021-2025": (1140, N)}
def mask(si, j, per):
    a, b = PER[per]; return (D["st"] == si) & (D["j"] == j) & (D["split"] == 2) & D["ok"] & (D["T"] >= a) & (D["T"] < b)

# RMSE and MAE (mm) per period, horizon, station, and method
rows = []
for per in PER:
    for j, h in enumerate(HORIZONS):
        for st in TESTSTS:
            k = mask(sts.index(st), j, per)
            if k.sum() < 12: continue
            for m in METHODS:
                e = (y[k] - PRED[m][k]) * SC[k]
                rows.append(dict(period=per, h=h, station=st, n=int(k.sum()), method=m,
                                 RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e)))))
df = pd.DataFrame(rows)
R = {"per_station": rows, "mean": {}, "wins_mae": {}, "panel": {}}
for per in PER:
    d = df[df.period == per]
    w = d.pivot_table(index=["h", "station"], columns="method", values="MAE")
    skill = 1 - w[METHODS].div(w["Clim-30yr"], axis=0)          # MAE skill per station, then averaged
    R["mean"][per] = {}
    for h in HORIZONS:
        dh = d[d.h == h]
        R["mean"][per][h] = {m: {"RMSE": float(dh[dh.method == m].RMSE.mean()), "MAE": float(dh[dh.method == m].MAE.mean()),
                                 "skill_MAE": float(skill.loc[h, m].mean()), "stations": int((dh.method == m).sum())} for m in METHODS}

PAIRS = [("GG3", "L-G3"), ("GG3", "NNLS"), ("GG3", "Mean"), ("GG3", "Clim-30yr"), ("GG4", "L-G4"),
         ("GG4-transfer", "Clim-30yr"), ("GG4-transfer", "GG4"), ("GG4-transfer", "L-G4"), ("GG4-transfer", "NNLS")]
for a, b in PAIRS:
    key = f"{a} vs {b}"; R["wins_mae"][key] = {}; R["panel"][key] = {}
    for per in PER:
        w = df[df.period == per].pivot_table(index=["h", "station"], columns="method", values="MAE")
        R["wins_mae"][key][per] = {"wins": int((w[a] < w[b]).sum()), "total": int(len(w)),
                                   "median_MAE_reduction": float(np.median(1 - w[a] / w[b]))}
        for loss in ["squared", "absolute"]:
            res = []
            for j, h in enumerate(HORIZONS):
                dd = {}
                for st in TESTSTS:
                    k = np.where(mask(sts.index(st), j, per))[0]
                    if len(k) < 12: continue
                    for kk in k:
                        ea, eb = y[kk] - PRED[a][kk], y[kk] - PRED[b][kk]   # errors scaled by the station standard deviation
                        dd.setdefault(int(D["T"][kk]), []).append(abs(ea) - abs(eb) if loss == "absolute" else ea**2 - eb**2)
                dser = np.array([np.mean(dd[t]) for t in sorted(dd)]); n = len(dser); dc = dser - dser.mean()
                gam = [np.sum(dc[l:] * dc[:n - l]) / n for l in range(h)]; v = gam[0] + 2 * sum(gam[1:]); v = v if v > 0 else gam[0]
                stat = dser.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
                res.append((h, float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1))), n))
            for (h, s_, p_, n), q in zip(res, holm([r[2] for r in res])):   # Holm across the 4 horizons
                R["panel"][key].setdefault(f"{per} h={h}", {})[loss] = {"DM": s_, "p": p_, "p_holm_horizon": float(q), "months": n}

with open(os.path.join(OUT, "error_metrics.json"), "w", newline="\n") as f:
    json.dump(R, f, indent=1)

print("== Mean RMSE and MAE across stations (mm), 2015-2025")
for h in HORIZONS:
    r = R["mean"]["2015-2025"][h]
    print(f"h={h:2d}  RMSE:", {m: round(r[m]["RMSE"], 2) for m in ["Clim-30yr", "Mean", "NNLS", "L-G3", "GG3", "GG4-transfer"]})
    print(f"      MAE :", {m: round(r[m]["MAE"], 2) for m in ["Clim-30yr", "Mean", "NNLS", "L-G3", "GG3", "GG4-transfer"]})
print("\n== Panel DM 2015-2025, Holm p (squared error | absolute error), h = 1, 3, 6, 12")
for key in R["panel"]:
    print(key, [round(R["panel"][key][f"2015-2025 h={h}"]["squared"]["p_holm_horizon"], 3) for h in HORIZONS], "|",
          [round(R["panel"][key][f"2015-2025 h={h}"]["absolute"]["p_holm_horizon"], 3) for h in HORIZONS])
