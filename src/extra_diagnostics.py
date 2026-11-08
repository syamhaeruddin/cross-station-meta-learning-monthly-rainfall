"""Diagnostics requested by the reviewers, all computed from the stored forecasts.

  A  transfer results per station and per leave-group-out fold
  B  climate-stratified diagnostics: wet vs dry season, and summer- vs winter-rainfall stations
  C  Diebold-Mariano bandwidth sensitivity and a moving-block bootstrap that preserves
     within-month cross-station dependence
  D  paired effect intervals for the mean scaled loss differential

Writes results/extra_diagnostics.json.
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from meta_gate import load_ds, station_nnls_prior, FOLDS

rng = np.random.default_rng(0)
D = load_ds()
sts = list(D["stations"]); pool = list(D["pool"]); M = len(pool); NB = M - 2
P, y, SC, T, J, ST, OK, SP = D["P"], D["y"], D["SC"], D["T"], D["j"], D["st"], D["ok"], D["split"]
TESTSTS = [s for s in sts if s not in ("Coffs_Harbour", "Melbourne")]

LP, W = station_nnls_prior(D)
PRED = {"NNLS": np.array([P[i] @ W[(ST[i], J[i])] for i in range(len(y))]),
        "Clim-30yr": P[:, pool.index("Clim-30yr")], "Mean11": P.mean(1), "Mean9": P[:, :NB].mean(1)}


def put(name, idx, arr):
    PRED.setdefault(name, np.full(len(y), np.nan))[idx] = arr.mean(0)


for f in glob.glob(os.path.join(OUT, "pred_local_*.npz")):
    Z = np.load(f)
    for nm in ["L-G3", "L-G4"]:
        put(nm, Z["idx"], Z[nm])
for nm in ["GG3", "GG4"]:
    f = os.path.join(OUT, f"pred_global_{nm}.npz")
    if os.path.exists(f):
        Z = np.load(f); put(nm, Z["idx"], Z["pred"])
FOLD_OF = {}
for fi in range(len(FOLDS)):
    f = os.path.join(OUT, f"pred_transfer_{fi}.npz")
    if os.path.exists(f):
        Z = np.load(f); put("GG4-T", Z["idx"], Z["pred"])
        for s_ in FOLDS[fi]:
            FOLD_OF[s_] = fi

R = {}


def m2(si, j):
    return (ST == si) & (J == j) & (SP == 2) & OK


# ---------------------------------------------------------------- A. transfer
print("== A. Transfer per station (NSE averaged over the four horizons, test 2015-2025)")
print(f"{'fold':>5} {'station':<16}{'GG4-T':>8}{'GG4':>8}{'L-G4':>8}{'Clim30':>8}{'dMSE vs L-G4':>14}{'wins/4':>8}")
A = {}
for fi in range(len(FOLDS)):
    for s_ in FOLDS[fi]:
        if s_ not in TESTSTS:
            continue
        si = sts.index(s_); vals = {m: [] for m in ["GG4-T", "GG4", "L-G4", "Clim-30yr"]}
        dm_, wins = [], 0
        for j, h in enumerate(HORIZONS):
            k = m2(si, j)
            if k.sum() < 12:
                continue
            yy = y[k] * SC[k]
            for m in vals:
                if not np.isnan(PRED[m][k]).any():
                    vals[m].append(metrics(yy, PRED[m][k] * SC[k])["NSE"])
            e1, e2 = y[k] - PRED["GG4-T"][k], y[k] - PRED["L-G4"][k]
            r_ = 1 - np.mean(e1 ** 2) / np.mean(e2 ** 2); dm_.append(float(r_)); wins += int(r_ > 0)
        a = {m: (float(np.mean(v)) if v else float("nan")) for m, v in vals.items()}
        a.update(fold=fi, median_dMSE_vs_LG4=float(np.median(dm_)) if dm_ else float("nan"), wins_vs_LG4=wins)
        A[s_] = a
        print(f"{fi:>5} {s_.replace('_',' '):<16}{a['GG4-T']:>8.3f}{a['GG4']:>8.3f}{a['L-G4']:>8.3f}{a['Clim-30yr']:>8.3f}{a['median_dMSE_vs_LG4']:>+14.3%}{wins:>6}/4")
R["transfer_per_station"] = A

print("\n== A2. Transfer per fold (mean over the stations of the fold)")
R["transfer_per_fold"] = {}
for fi in range(len(FOLDS)):
    mem = [s_ for s_ in FOLDS[fi] if s_ in A]
    if not mem:
        continue
    g = {m: float(np.mean([A[s_][m] for s_ in mem])) for m in ["GG4-T", "GG4", "L-G4", "Clim-30yr"]}
    g["stations"] = [s_.replace("_", " ") for s_ in mem]
    g["median_dMSE_vs_LG4"] = float(np.mean([A[s_]["median_dMSE_vs_LG4"] for s_ in mem]))
    R["transfer_per_fold"][fi] = g
    print(f"  fold {fi} {', '.join(g['stations']):<40} GG4-T {g['GG4-T']:.3f}  L-G4 {g['L-G4']:.3f}  dMSE {g['median_dMSE_vs_LG4']:+.2%}")

# ------------------------------------------------------- B. stratifikasi iklim
print("\n== B. Climate stratification")
WET, REGIME = {}, {}
for s_ in TESTSTS:
    yv, okv = load(s_); mon = (np.arange(N) % 12) + 1
    clim = np.array([yv[:VA0][okv[:VA0] & (mon[:VA0] == m)].mean() for m in range(1, 13)])
    WET[s_] = set((np.argsort(clim)[-6:] + 1).tolist())
    REGIME[s_] = "summer" if int(np.argmax(clim)) + 1 in (11, 12, 1, 2, 3) else "winter"
print("  summer-rainfall:", ", ".join(s_.replace("_", " ") for s_ in TESTSTS if REGIME[s_] == "summer"))
print("  winter-rainfall:", ", ".join(s_.replace("_", " ") for s_ in TESTSTS if REGIME[s_] == "winter"))

R["stratified"] = {}
for pair in [("GG3", "L-G3"), ("GG3", "Mean11"), ("GG3", "NNLS"), ("GG4-T", "L-G4")]:
    a, b = pair; key = f"{a} vs {b}"; R["stratified"][key] = {}
    print(f"\n  {key}: median change in MSE (negative = {a} better)")
    for grp, sel in [("all stations", lambda s_: True),
                     ("summer-rainfall", lambda s_: REGIME[s_] == "summer"),
                     ("winter-rainfall", lambda s_: REGIME[s_] == "winter")]:
        for seas in ["wet months", "dry months", "both"]:
            vals = []
            for s_ in TESTSTS:
                if not sel(s_):
                    continue
                si = sts.index(s_)
                for j, h in enumerate(HORIZONS):
                    k = m2(si, j)
                    if seas != "both":
                        mm = (T % 12) + 1
                        inw = np.isin(mm, list(WET[s_]))
                        k = k & (inw if seas == "wet months" else ~inw)
                    if k.sum() < 12 or np.isnan(PRED[a][k]).any() or np.isnan(PRED[b][k]).any():
                        continue
                    e1, e2 = y[k] - PRED[a][k], y[k] - PRED[b][k]
                    vals.append(float(np.mean(e1 ** 2) / np.mean(e2 ** 2) - 1))
            if vals:
                R["stratified"][key][f"{grp} / {seas}"] = {"median": float(np.median(vals)), "n": len(vals),
                                                           "better": int(sum(v < 0 for v in vals))}
                print(f"    {grp:<16}{seas:<12}{np.median(vals):+8.2%}   better in {sum(v < 0 for v in vals)}/{len(vals)}")

# ------------------------------- C/D. bandwidth, block bootstrap, interval
def panel_series(a, b, j):
    dd = {}
    for s_ in TESTSTS:
        si = sts.index(s_); k = np.where(m2(si, j))[0]
        if len(k) < 12 or np.isnan(PRED[a][k]).any() or np.isnan(PRED[b][k]).any():
            continue
        for kk in k:
            dd.setdefault(int(T[kk]), []).append((y[kk] - PRED[a][kk]) ** 2 - (y[kk] - PRED[b][kk]) ** 2)
    Ts = sorted(dd)
    return np.array([np.mean(dd[t]) for t in Ts])


def dm_lag(d, L):
    n = len(d); dc = d - d.mean()
    gam = [np.sum(dc[l:] * dc[:n - l]) / n for l in range(max(L, 1))]
    v = gam[0] + 2 * sum(gam[1:]); v = v if v > 0 else gam[0]
    stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n) if L >= 1 else d.mean() / np.sqrt(v / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


def block_boot(d, bl=12, B=4000):
    n = len(d); nb = int(np.ceil(n / bl)); mu = d.mean(); dc = d - mu
    starts = rng.integers(0, n - bl + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(bl)[None, None, :]).reshape(B, -1)[:, :n]
    bm = dc[idx].mean(1)
    p = float((np.abs(bm) >= abs(mu)).mean())
    lo, hi = np.percentile(mu - bm, [2.5, 97.5])
    return p, float(lo), float(hi)


print("\n== C/D. Bandwidth sensitivity, moving-block bootstrap and 95% intervals")
R["robust"] = {}
for a, b in [("GG3", "L-G3"), ("GG3", "NNLS"), ("GG3", "Clim-30yr"), ("GG3", "Mean11"), ("GG4-T", "GG4")]:
    key = f"{a} vs {b}"; R["robust"][key] = {}
    print(f"\n  {key}")
    print(f"    {'h':>3}{'months':>8}{'p(h-1)':>9}{'p(L=6)':>9}{'p(L=12)':>10}{'p(auto)':>10}{'p(block)':>10}   95% CI of mean scaled dMSE")
    for j, h in enumerate(HORIZONS):
        d = panel_series(a, b, j)
        if len(d) < 12:
            continue
        n = len(d); auto = int(np.floor(4 * (n / 100) ** (2 / 9)))
        res = {}
        for lab, L in [("h-1", h - 1), ("L=6", 6), ("L=12", 12), ("auto", max(auto, 1))]:
            st_, p_ = dm_lag(d, max(L, 1) if L >= 1 else 1)
            res[lab] = {"DM": st_, "p": p_, "lag": int(max(L, 1))}
        pb, lo, hi = block_boot(d)
        res["block_bootstrap_p"] = pb; res["ci95"] = [lo, hi]; res["months"] = n; res["mean"] = float(d.mean())
        R["robust"][key][h] = res
        print(f"    {h:>3}{n:>8}{res['h-1']['p']:>9.3f}{res['L=6']['p']:>9.3f}{res['L=12']['p']:>10.3f}{res['auto']['p']:>10.3f}{pb:>10.3f}   [{lo:+.4f}, {hi:+.4f}]")

json.dump(R, open(os.path.join(OUT, "extra_diagnostics.json"), "w", newline="\n"), indent=1)
print("\nwritten results/extra_diagnostics.json")
