"""Controls requested by the reviewers, all fitted on pre-test observed months only
and evaluated with the conventions of src/evaluate.py.

  Mean11      equal weights over all eleven members, including both climatologies
  NNLS-month  fixed month-specific weights, per station, horizon and calendar month
  Cal-global  calendar-only global combiner: one weight vector per horizon and month,
              pooled across stations, with no error history and no regime input
  NNLS-HG     hard global NNLS: one weight vector per horizon pooled across stations
  NNLS-SG     soft global NNLS: per-station weights shrunk toward a shared vector,
              the penalty chosen on the validation split

Writes results/extra_baselines.json.
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from meta_gate import load_ds, station_nnls_prior
from scipy.optimize import nnls

D = load_ds()
sts = list(D["stations"]); pool = list(D["pool"]); M = len(pool); NB = M - 2
P, y, SC, T, J, ST, OK, SP = D["P"], D["y"], D["SC"], D["T"], D["j"], D["st"], D["ok"], D["split"]
MONTH = (T % 12) + 1
TESTSTS = [s for s in sts if s not in ("Coffs_Harbour", "Melbourne")]
PRED = {}


def fit_nnls(k):
    if k.sum() < M + 2:
        return None
    w, _ = nnls(P[k], y[k])
    return w if w.sum() > 1e-9 else None


# --- 1. equal weights over all eleven members -------------------------------
PRED["Mean11"] = P.mean(1)
PRED["Mean9"] = P[:, :NB].mean(1)

# --- 2. fixed month-specific weights, per station and horizon ---------------
pm = np.full(len(y), np.nan)
for si in range(len(sts)):
    for j in range(4):
        for mo in range(1, 13):
            k = (ST == si) & (J == j) & (MONTH == mo)
            w = fit_nnls(k & (SP < 2) & OK)
            if w is None:
                w = fit_nnls((ST == si) & (J == j) & (SP < 2) & OK)      # fall back to all months
            if w is not None:
                pm[k] = P[k] @ w
PRED["NNLS-month"] = pm

# --- 3. calendar-only global combiner ---------------------------------------
cg = np.full(len(y), np.nan)
for j in range(4):
    for mo in range(1, 13):
        k = (J == j) & (MONTH == mo)
        w = fit_nnls(k & (SP < 2) & OK)
        if w is not None:
            cg[k] = P[k] @ w
PRED["Cal-global"] = cg

# --- 4. hard global NNLS -----------------------------------------------------
hg = np.full(len(y), np.nan); WHG = {}
for j in range(4):
    k = J == j
    w = fit_nnls(k & (SP < 2) & OK); WHG[j] = w
    hg[k] = P[k] @ w
PRED["NNLS-HG"] = hg


# --- 5. soft global NNLS ------------------------------------------------------
def soft_global(j, g, train):
    """Per-station weights shrunk toward a shared vector with penalty g."""
    k0 = (J == j) & train
    wbar = fit_nnls(k0)
    I = np.eye(M); r = np.sqrt(g)
    for _ in range(25):
        Ws = {}
        for si in range(len(sts)):
            k = (ST == si) & k0
            if k.sum() < 4:
                Ws[si] = wbar.copy(); continue
            A = np.vstack([P[k], r * I]); b = np.concatenate([y[k], r * wbar])
            Ws[si], _ = nnls(A, b)
        new = np.mean([Ws[s] for s in Ws], 0)
        if np.max(np.abs(new - wbar)) < 1e-6:
            wbar = new; break
        wbar = new
    return Ws


GAMMAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
best_g = {}
for j in range(4):
    tr = (SP == 0) & OK; va = (J == j) & (SP == 1) & OK
    scores = []
    for g in GAMMAS:
        Ws = soft_global(j, g, tr) if g > 0 else {si: (fit_nnls((ST == si) & (J == j) & tr) if ((ST == si) & (J == j) & tr).sum() >= M + 2 else fit_nnls((J == j) & tr)) for si in range(len(sts))}
        pv = np.array([P[i] @ (Ws[ST[i]] if Ws[ST[i]] is not None else WHG[j]) for i in np.where(va)[0]])
        scores.append(float(np.mean((y[va] - pv) ** 2)))
    best_g[j] = GAMMAS[int(np.argmin(scores))]
    print(f"h={HORIZONS[j]}  validation MSE by gamma:", [round(s, 4) for s in scores], "-> gamma =", best_g[j], flush=True)

sg = np.full(len(y), np.nan)
for j in range(4):
    Ws = soft_global(j, best_g[j], (SP < 2) & OK)
    for si in range(len(sts)):
        k = (ST == si) & (J == j)
        w = Ws.get(si)
        sg[k] = P[k] @ (w if w is not None else WHG[j])
PRED["NNLS-SG"] = sg

# --- bring in the published methods for comparison ---------------------------
LP, W = station_nnls_prior(D)
PRED["NNLS"] = np.array([P[i] @ W[(ST[i], J[i])] for i in range(len(y))])
PRED["Clim-30yr"] = P[:, pool.index("Clim-30yr")]


def put(name, idx, arr):
    full = PRED.setdefault(name, np.full(len(y), np.nan)); full[idx] = arr.mean(0)


for f in glob.glob(os.path.join(OUT, "pred_local_*.npz")):
    Z = np.load(f); put("L-G3", Z["idx"], Z["L-G3"])
for nm in ["GG3", "GG4"]:
    f = os.path.join(OUT, f"pred_global_{nm}.npz")
    if os.path.exists(f):
        Z = np.load(f); put(nm, Z["idx"], Z["pred"])

NEW = ["Mean9", "Mean11", "NNLS-month", "Cal-global", "NNLS-HG", "NNLS-SG", "NNLS", "Clim-30yr", "L-G3", "GG3"]


def mask(si, j):
    return (ST == si) & (J == j) & (SP == 2) & OK


R = {"gamma": {HORIZONS[j]: best_g[j] for j in range(4)}, "nse_mean": {}, "panel": {}}
print("\n== Mean NSE across the 13 evaluated stations, test 2015-2025")
hdr = f"{'h':>4}" + "".join(f"{m:>12}" for m in NEW)
print(hdr)
for j, h in enumerate(HORIZONS):
    row = {}
    for m in NEW:
        vals = []
        for st in TESTSTS:
            k = mask(sts.index(st), j)
            if k.sum() < 12 or np.isnan(PRED[m][k]).any():
                continue
            vals.append(metrics(y[k] * SC[k], PRED[m][k] * SC[k])["NSE"])
        row[m] = float(np.mean(vals)) if vals else float("nan")
    R["nse_mean"][h] = row
    print(f"{h:>4}" + "".join(f"{row[m]:>12.3f}" for m in NEW))

PAIRS = [("GG3", "Mean11"), ("GG3", "NNLS-HG"), ("GG3", "NNLS-SG"), ("GG3", "Cal-global"),
         ("GG3", "NNLS-month"), ("NNLS-SG", "NNLS"), ("NNLS-HG", "NNLS"), ("Mean11", "Mean9")]
print("\n== Panel DM, squared error, test 2015-2025 (negative favours the first method)")
for a, b in PAIRS:
    key = f"{a} vs {b}"; R["panel"][key] = {}; out = []
    ps = []
    for j, h in enumerate(HORIZONS):
        dd = {}
        for st in TESTSTS:
            si = sts.index(st); k = np.where(mask(si, j))[0]
            if len(k) < 12 or np.isnan(PRED[a][k]).any() or np.isnan(PRED[b][k]).any():
                continue
            for kk in k:
                dd.setdefault(int(T[kk]), []).append((y[kk] - PRED[a][kk]) ** 2 - (y[kk] - PRED[b][kk]) ** 2)
        Ts = sorted(dd); d = np.array([np.mean(dd[t]) for t in Ts])
        if len(d) < 12:
            continue
        n = len(d); dc = d - d.mean(); gam = [np.sum(dc[l:] * dc[:n - l]) / n for l in range(h)]
        v = gam[0] + 2 * sum(gam[1:]); v = v if v > 0 else gam[0]
        stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p = float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))
        R["panel"][key][h] = {"DM": float(stat), "p": p, "months": n}; ps.append((h, p))
    for (h, _), q in zip(ps, holm([p for _, p in ps])):
        R["panel"][key][h]["p_holm"] = float(q)
    print(f"  {key:<24}" + "  ".join(f"h={h}: DM {R['panel'][key][h]['DM']:+.2f} p {R['panel'][key][h]['p_holm']:.3f}" for h in R["panel"][key]))

json.dump(R, open(os.path.join(OUT, "extra_baselines.json"), "w", newline="\n"), indent=1)
print("\nwritten results/extra_baselines.json")
