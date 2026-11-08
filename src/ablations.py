"""Controlled ablations requested by Reviewer 2, separating sample pooling from the
station embedding, the static descriptors, the batch size and the error history.

  python src/ablations.py main    -> GG-P, GG-S, GG3-b64, GG-noE   (writes results/pred_abl_*.npz)
  python src/ablations.py curve   -> station-subsampling learning curve
  python src/ablations.py eval    -> mean NSE and panel DM for everything produced above
"""
import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from meta_gate import load_ds, station_nnls_prior, fit_predict, FOLDS
from scipy.optimize import nnls

cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
D = load_ds(); LP, W = station_nnls_prior(D)
sts = list(D["stations"]); pool = list(D["pool"]); M = len(pool); NB = M - 2
P, y, SC, T, J, ST, OK, SP = D["P"], D["y"], D["SC"], D["T"], D["j"], D["st"], D["ok"], D["split"]
TESTSTS = [s for s in sts if s not in ("Coffs_Harbour", "Melbourne")]
base_tr = (SP == 0) & OK; base_va = (SP == 1) & OK
te_all = np.where(SP == 2)[0]
HOLD = ["Darwin", "Perth", "Woomera"]            # data-rich holdout
HOLD2 = ["Adelaide", "Birdsville", "Innisfail"]  # data-poor holdout: the shortest observed records
CURVES = {"curve": (HOLD, "pred_curve_"), "curve2": (HOLD2, "pred_curve2_")}


def prior_from(mask_tr):
    """Pooled NNLS prior per horizon from the given training rows only."""
    LPx = np.zeros_like(P)
    for j in range(4):
        k = mask_tr & (J == j)
        w, _ = nnls(P[k], y[k])
        w = np.maximum(w / max(w.sum(), 1e-9), 0.01)
        LPx[J == j] = np.log(w / w.sum())
    return LPx


if cmd == "main":
    t0 = time.time()
    VARIANTS = [
        # tag        anchor  emb    static  batch  zero_E
        ("GG-P",     False,  False, False,  256,   False),   # pooling only: local inputs, global samples
        ("GG-S",     False,  False, True,   256,   False),   # pooling + static, no embedding
        ("GG3-b64",  False,  True,  True,   64,    False),   # full global gate at the local batch size
        ("GG-noE",   False,  True,  True,   256,   True),    # full global gate with the error history removed
    ]
    for tag, anc, emb, stat, batch, zeroE in VARIANTS:
        if os.path.exists(os.path.join(OUT, f"pred_abl_{tag}.npz")):
            print(tag, "already present, skipped", flush=True); continue
        Dx = D
        if zeroE:
            Dx = dict(D); Dx["E"] = np.zeros_like(D["E"])
        p, w = fit_predict(Dx, np.where(base_tr)[0], np.where(base_va)[0], te_all, anc, emb, stat, LP, batch, tag)
        np.savez(os.path.join(OUT, f"pred_abl_{tag}.npz"), idx=te_all, pred=p)
        print(tag, "done", round((time.time() - t0) / 60, 1), "min", flush=True)

elif cmd in CURVES:
    HOLDX, PREFIX = CURVES[cmd]
    t0 = time.time()
    rng = np.random.default_rng(7)
    hold_i = [sts.index(s) for s in HOLDX]
    pool_st = [i for i in range(len(sts)) if i not in hold_i and sts[i] in TESTSTS]
    inh = np.isin(ST, hold_i)
    te = np.where(inh & (SP == 2))[0]
    KS = [2, 4, 6, 9]
    DRAWS = 4
    out = {}
    for k in KS:
        for dnum in range(DRAWS if k < len(pool_st) else 1):
            pick = sorted(rng.choice(pool_st, size=min(k, len(pool_st)), replace=False).tolist())
            sel = np.isin(ST, pick)
            LPx = prior_from(sel & base_tr)
            tag = f"k{k}_d{dnum}"
            p, w = fit_predict(D, np.where(sel & base_tr)[0], np.where(sel & base_va)[0], te,
                               True, False, True, LPx, 256, tag)
            np.savez(os.path.join(OUT, f"{PREFIX}{tag}.npz"), idx=te, pred=p,
                     train_stations=np.array([sts[i] for i in pick]))
            out[tag] = [sts[i] for i in pick]
            print(tag, [sts[i] for i in pick], round((time.time() - t0) / 60, 1), "min", flush=True)
    # all remaining stations
    sel = np.isin(ST, pool_st)
    LPx = prior_from(sel & base_tr)
    p, w = fit_predict(D, np.where(sel & base_tr)[0], np.where(sel & base_va)[0], te, True, False, True, LPx, 256, "kall")
    np.savez(os.path.join(OUT, f"{PREFIX}kall_d0.npz"), idx=te, pred=p,
             train_stations=np.array([sts[i] for i in pool_st]))
    print("kall done", round((time.time() - t0) / 60, 1), "min", flush=True)

else:
    PRED = {"NNLS": np.array([P[i] @ W[(ST[i], J[i])] for i in range(len(y))]),
            "Clim-30yr": P[:, pool.index("Clim-30yr")], "Mean11": P.mean(1)}

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
    for f in sorted(glob.glob(os.path.join(OUT, "pred_abl_*.npz"))):
        Z = np.load(f); put(os.path.basename(f)[9:-4], Z["idx"], Z["pred"])

    ABL = [m for m in ["L-G3", "GG-P", "GG-S", "GG3-b64", "GG-noE", "GG3", "Mean11", "NNLS"] if m in PRED]
    R = {"nse_mean": {}, "panel": {}, "curve": {}}

    def m2(si, j):
        return (ST == si) & (J == j) & (SP == 2) & OK

    print("== Ablations: mean NSE across the 13 evaluated stations, test 2015-2025")
    print(f"{'h':>4}" + "".join(f"{m:>11}" for m in ABL))
    for j, h in enumerate(HORIZONS):
        row = {}
        for m in ABL:
            v = []
            for s_ in TESTSTS:
                k = m2(sts.index(s_), j)
                if k.sum() >= 12 and not np.isnan(PRED[m][k]).any():
                    v.append(metrics(y[k] * SC[k], PRED[m][k] * SC[k])["NSE"])
            row[m] = float(np.mean(v)) if v else float("nan")
        R["nse_mean"][h] = row
        print(f"{h:>4}" + "".join(f"{row[m]:>11.3f}" for m in ABL))

    def panel(a, b, stations=None):
        res = {}; ps = []
        for j, h in enumerate(HORIZONS):
            dd = {}
            for s_ in (stations or TESTSTS):
                si = sts.index(s_); k = np.where(m2(si, j))[0]
                if len(k) < 12 or np.isnan(PRED[a][k]).any() or np.isnan(PRED[b][k]).any():
                    continue
                for kk in k:
                    dd.setdefault(int(T[kk]), []).append((y[kk] - PRED[a][kk]) ** 2 - (y[kk] - PRED[b][kk]) ** 2)
            Ts = sorted(dd); d = np.array([np.mean(dd[t]) for t in Ts])
            if len(d) < 12:
                continue
            n = len(d); dc = d - d.mean(); gam = [np.sum(dc[l:] * dc[:n - l]) / n for l in range(h)]
            v = gam[0] + 2 * sum(gam[1:]); v = v if v > 0 else gam[0]
            s_stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
            p_ = float(2 * (1 - stats.t.cdf(abs(s_stat), df=n - 1)))
            res[h] = {"DM": float(s_stat), "p": p_}; ps.append((h, p_))
        for (h, _), q in zip(ps, holm([p for _, p in ps])):
            res[h]["p_holm"] = float(q)
        return res

    print("\n== Panel DM, squared error (negative favours the first method)")
    for a, b in [("GG-P", "L-G3"), ("GG-S", "GG-P"), ("GG3", "GG-P"), ("GG3", "GG-S"),
                 ("GG3", "GG3-b64"), ("GG3", "GG-noE"), ("GG-P", "Mean11")]:
        if a not in PRED or b not in PRED:
            continue
        r = panel(a, b); R["panel"][f"{a} vs {b}"] = r
        print(f"  {a:>8} vs {b:<9}" + "  ".join(f"h={h}: DM {r[h]['DM']:+.2f} p {r[h]['p_holm']:.3f}" for h in r))

    # learning curve
    for cname, (HOLDX, PREFIX) in CURVES.items():
      cf = sorted(glob.glob(os.path.join(OUT, PREFIX + "*.npz")))
      if cf:
        HOLD = HOLDX
        print("\n== Learning curve, holdout:", ", ".join(s_.replace("_", " ") for s_ in HOLD))
        print(f"{'k':>5}{'draws':>7}{'mean NSE':>11}{'sd':>8}")
        byk = {}
        for f in cf:
            tag = os.path.basename(f)[len(PREFIX):-4]; k = tag.split("_")[0][1:]
            Z = np.load(f, allow_pickle=True)
            pr = np.full(len(y), np.nan); pr[Z["idx"]] = Z["pred"].mean(0)
            vals = []
            for s_ in HOLD:
                si = sts.index(s_)
                for j, h in enumerate(HORIZONS):
                    kk = m2(si, j)
                    if kk.sum() >= 12 and not np.isnan(pr[kk]).any():
                        vals.append(metrics(y[kk] * SC[kk], pr[kk] * SC[kk])["NSE"])
            byk.setdefault(k, []).append(float(np.mean(vals)))
        for k in sorted(byk, key=lambda z: (z == "all", int(z) if z != "all" else 0)):
            v = byk[k]
            print(f"{k:>5}{len(v):>7}{np.mean(v):>11.3f}{(np.std(v, ddof=1) if len(v) > 1 else 0):>8.3f}")
            R.setdefault("curve_" + cname, {})[k] = {"draws": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
        for ref in ["L-G3", "L-G4", "GG3", "NNLS"]:
            if ref in PRED:
                vals = []
                for s_ in HOLD:
                    si = sts.index(s_)
                    for j, h in enumerate(HORIZONS):
                        kk = m2(si, j)
                        if kk.sum() >= 12 and not np.isnan(PRED[ref][kk]).any():
                            vals.append(metrics(y[kk] * SC[kk], PRED[ref][kk] * SC[kk])["NSE"])
                R.setdefault("curve_" + cname, {})["ref " + ref] = float(np.mean(vals))
                print(f"  reference {ref:<6} {np.mean(vals):.3f}")

    json.dump(R, open(os.path.join(OUT, "ablations.json"), "w", newline="\n"), indent=1)
    print("\nwritten results/ablations.json")
