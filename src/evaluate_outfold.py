# Comparison of the leave-group-out transfer gate under two preprocessing choices:
#   GG4-T        static descriptors standardized once over all fifteen stations (results/pred_transfer_<f>.npz)
#   GG4-T-of     standardization refitted on the twelve training stations of each fold (results/pred_transfer_of_<f>.npz)
# Reports per-horizon and per-fold NSE on the held-out stations and panel DM tests against
# GG4, L-G4, NNLS and Clim-30yr. Writes results/evaluation_outfold.json. Reads only; it does
# not touch evaluation_results.json.
#
#   python src/evaluate_outfold.py
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from meta_gate import load_ds, station_nnls_prior, FOLDS

D = load_ds(); LP, W = station_nnls_prior(D)
sts = list(D["stations"]); pool = list(D["pool"]); M = len(pool); NB = M - 2
P, y, SC = D["P"], D["y"], D["SC"]
iK30 = pool.index("Clim-30yr")

PRED = {"Clim-30yr": P[:, iK30], "Mean": P[:, :NB].mean(1),
        "NNLS": np.array([P[k] @ W[(D["st"][k], D["j"][k])] for k in range(len(y))])}


def put(name, idx, arr):
    full = PRED.setdefault(name, np.full(len(y), np.nan)); full[idx] = arr.mean(0)


for f in glob.glob(os.path.join(OUT, "pred_local_*.npz")):
    Z = np.load(f)
    for nm in ["L-G3", "L-G4"]: put(nm, Z["idx"], Z[nm])
for nm in ["GG3", "GG4"]:
    fp = os.path.join(OUT, f"pred_global_{nm}.npz")
    if os.path.exists(fp): Z = np.load(fp); put(nm, Z["idx"], Z["pred"])

# the two transfer variants, kept apart: the original files are pred_transfer_<f>.npz,
# the out-of-fold ones pred_transfer_of_<f>.npz, so match the fold index explicitly
FOLDOF = {}      # station index -> fold, for the stations that are held out somewhere
for f in range(len(FOLDS)):
    for s in FOLDS[f]:
        if s in sts: FOLDOF[sts.index(s)] = f
for f in range(len(FOLDS)):
    for tag, fn in [("GG4-T", f"pred_transfer_{f}.npz"), ("GG4-T-of", f"pred_transfer_of_{f}.npz")]:
        fp = os.path.join(OUT, fn)
        if os.path.exists(fp): Z = np.load(fp); put(tag, Z["idx"], Z["pred"])
    if not os.path.exists(os.path.join(OUT, f"pred_transfer_of_{f}.npz")):
        print(f"  !! fold {f}: out-of-fold prediction file missing", flush=True)

METHODS = [m for m in ["Clim-30yr", "Mean", "NNLS", "L-G3", "L-G4", "GG3", "GG4", "GG4-T", "GG4-T-of"] if m in PRED]
TESTSTS = [s for s in sts if s not in ("Coffs_Harbour", "Melbourne")]
PER = {"2015-2025": (TE0, N), "2015-2020": (TE0, 1140), "2021-2025": (1140, N)}


def mask(si, j, per):
    a, b = PER[per]
    return (D["st"] == si) & (D["j"] == j) & (D["split"] == 2) & D["ok"] & (D["T"] >= a) & (D["T"] < b)


R = {"methods": METHODS, "folds": {str(f): FOLDS[f] for f in range(len(FOLDS))},
     "nse": {}, "fold_nse": {}, "panel": {}, "weights": {}}

# ---- per station-horizon NSE, held-out stations only --------------------------------
HOLD = [s for s in TESTSTS if sts.index(s) in FOLDOF]
for per in PER:
    R["nse"][per] = {}
    for st in HOLD:
        si = sts.index(st); R["nse"][per][st] = {"fold": FOLDOF[si]}
        for j, h in enumerate(HORIZONS):
            k = mask(si, j, per)
            if k.sum() < 12: continue
            yy = y[k] * SC[k]; r = {"n": int(k.sum())}
            for m in METHODS:
                if np.isnan(PRED[m][k]).any(): continue
                r[m] = round(metrics(yy, PRED[m][k] * SC[k])["NSE"], 4)
            R["nse"][per][st][h] = r

# ---- means by horizon and by fold, over the held-out stations -----------------------
for per in PER:
    R["fold_nse"][per] = {"by_horizon": {}, "by_fold": {}, "overall": {}}
    for m in METHODS:
        for j, h in enumerate(HORIZONS):
            v = [R["nse"][per][st][h][m] for st in HOLD
                 if h in R["nse"][per][st] and m in R["nse"][per][st][h]]
            if v: R["fold_nse"][per]["by_horizon"].setdefault(f"h={h}", {})[m] = round(float(np.mean(v)), 4)
        for f in range(len(FOLDS)):
            v = [R["nse"][per][st][h][m] for st in HOLD if FOLDOF[sts.index(st)] == f
                 for h in HORIZONS if h in R["nse"][per][st] and m in R["nse"][per][st][h]]
            if v: R["fold_nse"][per]["by_fold"].setdefault(f"fold {f}", {})[m] = round(float(np.mean(v)), 4)
        v = [R["nse"][per][st][h][m] for st in HOLD
             for h in HORIZONS if h in R["nse"][per][st] and m in R["nse"][per][st][h]]
        if v: R["fold_nse"][per]["overall"][m] = round(float(np.mean(v)), 4)

# ---- panel DM tests on the held-out stations ---------------------------------------
PAIRS = [(a, b) for a in ["GG4-T-of", "GG4-T"] if a in PRED
         for b in ["GG4-T", "GG4", "L-G4", "NNLS", "Clim-30yr"] if b in PRED and b != a]
PAIRS = [p for p in PAIRS if p != ("GG4-T", "GG4-T")]
for a, b in PAIRS:
    key = f"{a} vs {b}"; R["panel"][key] = {}
    for per in PER:
        for j, h in enumerate(HORIZONS):
            dd = {}
            for st in HOLD:
                si = sts.index(st); k = np.where(mask(si, j, per))[0]
                if len(k) < 12 or np.isnan(PRED[a][k]).any() or np.isnan(PRED[b][k]).any(): continue
                for kk in k:
                    dd.setdefault(int(D["T"][kk]), []).append((y[kk] - PRED[a][kk])**2 - (y[kk] - PRED[b][kk])**2)
            Ts = sorted(dd); d = np.array([np.mean(dd[t]) for t in Ts])
            if len(d) < 12: continue
            n = len(d); dc = d - d.mean(); gam = [np.sum(dc[l:] * dc[:n - l]) / n for l in range(h)]
            v = gam[0] + 2 * sum(gam[1:]); v = v if v > 0 else gam[0]
            stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
            R["panel"][key][f"{per} h={h}"] = {"DM": round(float(stat), 3),
                                               "p": round(float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1))), 4),
                                               "months": n, "mean_dMSE_scaled": round(float(d.mean()), 5)}
        ks = [kk for kk in R["panel"][key] if kk.startswith(per)]
        if ks:
            for kk, q in zip(ks, holm([R["panel"][key][kk]["p"] for kk in ks])):
                R["panel"][key][kk]["p_holm_horizon"] = round(float(q), 4)

# ---- mean gate weights, for the record ---------------------------------------------
for f in range(len(FOLDS)):
    for tag, fn in [("GG4-T", f"pred_transfer_{f}.npz"), ("GG4-T-of", f"pred_transfer_of_{f}.npz")]:
        fp = os.path.join(OUT, fn)
        if os.path.exists(fp):
            Z = np.load(fp)
            if "w" in Z: R["weights"].setdefault(tag, {})[f"fold {f}"] = [round(float(x), 4) for x in np.asarray(Z["w"]).mean(0).ravel()[:M]]

with open(os.path.join(OUT, "evaluation_outfold.json"), "w") as fh:
    json.dump(R, fh, indent=1)

# ---- console summary ---------------------------------------------------------------
per = "2015-2025"
print("\n=== held-out stations, NSE, test period " + per + " ===")
hdr = [m for m in ["Clim-30yr", "NNLS", "L-G4", "GG4", "GG4-T", "GG4-T-of"] if m in METHODS]
print("by horizon  " + "".join(f"{m:>11}" for m in hdr))
for j, h in enumerate(HORIZONS):
    row = R["fold_nse"][per]["by_horizon"].get(f"h={h}", {})
    print(f"  h={h:<9}" + "".join(f"{row.get(m, float('nan')):>11.4f}" for m in hdr))
print("by fold")
for f in range(len(FOLDS)):
    row = R["fold_nse"][per]["by_fold"].get(f"fold {f}", {})
    nm = ", ".join(FOLDS[f])
    print(f"  fold {f:<7}" + "".join(f"{row.get(m, float('nan')):>11.4f}" for m in hdr) + f"   [{nm}]")
row = R["fold_nse"][per]["overall"]
print("  overall  " + "".join(f"{row.get(m, float('nan')):>11.4f}" for m in hdr))

print("\n=== panel DM, held-out stations, " + per + " (Holm across the four horizons) ===")
for key in R["panel"]:
    cells = [f"h={h}: DM {R['panel'][key][f'{per} h={h}']['DM']:+.2f} p {R['panel'][key][f'{per} h={h}']['p_holm_horizon']:.3f}"
             for h in HORIZONS if f"{per} h={h}" in R["panel"][key]]
    print(f"  {key:<24} " + " | ".join(cells))

print("\n=== per station, NSE, " + per + " ===")
for st in HOLD:
    for h in HORIZONS:
        r = R["nse"][per][st].get(h)
        if not r: continue
        print(f"  {st:<16} h={h:<3}" + "".join(f"{r.get(m, float('nan')):>11.4f}" for m in hdr))
print("\nwrote results/evaluation_outfold.json")
