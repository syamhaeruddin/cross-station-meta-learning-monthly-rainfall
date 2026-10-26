# Tahap 5: evaluasi gate lokal, global, dan transfer terhadap pembanding statis; uji DM per stasiun (Holm) dan DM panel;
# diagnosis kelayakan ensemble. Menulis hasil/hasil_evaluasi.json.
#   python src/eval_global.py
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from meta_global import load_ds, station_nnls_prior, FOLDS

D = load_ds(); LP, W = station_nnls_prior(D)
sts = list(D["stations"]); pool = list(D["pool"]); M = len(pool); NB = M - 2
te = np.where(D["split"] == 2)[0]; P, y, SC = D["P"], D["y"], D["SC"]
iK30, iKE = pool.index("Klim-30th"), pool.index("Klim-ekspansif")

PRED = {"Klim-30th": P[:, iK30], "Klim-ekspansif": P[:, iKE], "Rata-rata": P[:, :NB].mean(1),
        "NNLS": np.array([P[k] @ W[(D["st"][k], D["j"][k])] for k in range(len(y))])}
# model dasar terbaik per stasiun-horizon menurut MSE pra-uji
best = np.zeros(len(y))
for si in range(len(sts)):
    for j in range(4):
        k = (D["st"] == si) & (D["j"] == j); kp = k & (D["split"] < 2) & D["ok"]
        i = int(np.argmin(((P[kp] - y[kp, None])**2).mean(0))); best[k] = P[k, i]
PRED["Model terbaik"] = best
SEEDSD = {}
def put(name, idx, arr):
    full = PRED.setdefault(name, np.full(len(y), np.nan)); full[idx] = arr.mean(0)
    sd = SEEDSD.setdefault(name, np.full((len(y), arr.shape[0]), np.nan)); sd[idx] = arr.T
for f in glob.glob(os.path.join(OUT, "pred_local_*.npz")):
    Z = np.load(f)
    for nm in ["L-G3", "L-G4"]: put(nm, Z["idx"], Z[nm])
for nm in ["GG3", "GG4"]:
    f = os.path.join(OUT, f"pred_global_{nm}.npz")
    if os.path.exists(f): Z = np.load(f); put(nm, Z["idx"], Z["pred"])
for f in glob.glob(os.path.join(OUT, "pred_transfer_*.npz")):
    Z = np.load(f); put("GG4-transfer", Z["idx"], Z["pred"])
METHODS = [m for m in ["Klim-30th", "Klim-ekspansif", "Rata-rata", "NNLS", "Model terbaik", "L-G3", "L-G4", "GG3", "GG4", "GG4-transfer"] if m in PRED]
TESTSTS = [s for s in sts if s not in ("Coffs_Harbour", "Melbourne")]
PER = {"2015-2025": (TE0, N), "2015-2020": (TE0, 1140), "2021-2025": (1140, N)}

def mask(si, j, per):
    a, b = PER[per]; return (D["st"] == si) & (D["j"] == j) & (D["split"] == 2) & D["ok"] & (D["T"] >= a) & (D["T"] < b)

R = {"stasiun": TESTSTS, "metode": METHODS, "nse": {}, "dm": {}, "panel": {}, "menang": {}, "diag": {}, "bobot": {}}
for per in PER:
    R["nse"][per] = {}
    for st in TESTSTS:
        si = sts.index(st); R["nse"][per][st] = {}
        for j, h in enumerate(HORIZONS):
            k = mask(si, j, per)
            if k.sum() < 12: continue
            yy = y[k] * SC[k]; r = {"n": int(k.sum())}
            for m in METHODS:
                if np.isnan(PRED[m][k]).any(): continue
                r[m] = metrics(yy, PRED[m][k] * SC[k])["NSE"]
                if m in SEEDSD: r[m + "_sd"] = float(np.std([metrics(yy, SEEDSD[m][k, s_] * SC[k])["NSE"] for s_ in range(SEEDSD[m].shape[1])], ddof=1))
            R["nse"][per][st][h] = r

PAIRS = [("GG4", "L-G4"), ("GG4", "NNLS"), ("GG4", "Rata-rata"), ("GG4", "Klim-30th"), ("GG3", "L-G3"),
         ("L-G4", "NNLS"), ("L-G4", "Klim-30th"), ("NNLS", "Klim-30th"), ("GG4-transfer", "NNLS"), ("GG4-transfer", "Klim-30th"), ("GG4-transfer", "GG4"),
         ("GG3", "NNLS"), ("GG3", "Rata-rata"), ("GG3", "Klim-30th"), ("GG3", "GG4"), ("GG4-transfer", "L-G4")]
PAIRS = [p for p in PAIRS if p[0] in PRED and p[1] in PRED]
for a, b in PAIRS:
    key = f"{a} vs {b}"; rows = []
    for st in TESTSTS:
        si = sts.index(st)
        for j, h in enumerate(HORIZONS):
            k = mask(si, j, "2015-2025")
            if k.sum() < 12 or np.isnan(PRED[a][k]).any() or np.isnan(PRED[b][k]).any(): continue
            e1, e2 = y[k] - PRED[a][k], y[k] - PRED[b][k]; t = dm_test(e1, e2, h)
            rows.append(dict(st=st, h=h, DM=t["DM"], p=t["p"], dMSE=float(1 - np.mean(e1**2) / np.mean(e2**2))))
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj): r["p_holm"] = float(q)
    R["dm"][key] = rows
    R["menang"][key] = {"lebih baik": int(sum(r["dMSE"] > 0 for r in rows)), "total": len(rows),
                        "signifikan lebih baik (Holm)": int(sum((r["DM"] < 0) & (r["p_holm"] < 0.05) for r in rows)),
                        "signifikan lebih buruk (Holm)": int(sum((r["DM"] > 0) & (r["p_holm"] < 0.05) for r in rows)),
                        "p uji tanda": float(stats.binomtest(int(sum(r["dMSE"] > 0 for r in rows)), len(rows), 0.5).pvalue) if rows else float("nan")}
    # DM panel: selisih galat kuadrat (skala per stasiun) dirata-rata lintas stasiun pada tiap bulan target
    R["panel"][key] = {}
    for per in PER:
        for j, h in enumerate(HORIZONS):
            dd = {}
            for st in TESTSTS:
                si = sts.index(st); k = np.where(mask(si, j, per))[0]
                if len(k) < 12 or np.isnan(PRED[a][k]).any() or np.isnan(PRED[b][k]).any(): continue
                for kk in k: dd.setdefault(int(D["T"][kk]), []).append((y[kk] - PRED[a][kk])**2 - (y[kk] - PRED[b][kk])**2)
            Ts = sorted(dd); d = np.array([np.mean(dd[t]) for t in Ts])
            if len(d) < 12: continue
            n = len(d); dc = d - d.mean(); gam = [np.sum(dc[l:] * dc[:n - l]) / n for l in range(h)]
            v = gam[0] + 2 * sum(gam[1:]); v = v if v > 0 else gam[0]
            stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
            R["panel"][key][f"{per} h={h}"] = {"DM": float(stat), "p": float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1))), "bulan": n,
                                               "rerata_dMSE_skala": float(d.mean())}
        ks = [kk for kk in R["panel"][key] if kk.startswith(per)]
        if ks:   # koreksi Holm lintas 4 horizon dalam satu perbandingan dan periode
            for kk, q in zip(ks, holm([R["panel"][key][kk]["p"] for kk in ks])): R["panel"][key][kk]["p_holm_horizon"] = float(q)

# Diagnosis kelayakan ensemble (data pra-uji 1962-2014, bulan teramati), lalu hubungannya dengan gain gate pada periode uji
for st in TESTSTS:
    si = sts.index(st); dg = {}
    for j, h in enumerate(HORIZONS):
        k = (D["st"] == si) & (D["j"] == j) & (D["split"] < 2) & D["ok"]
        e = y[k, None] - P[k]; eb = e[:, :NB]
        C = np.corrcoef(eb.T); corr = float(C[np.triu_indices(NB, 1)].mean())
        eq = P[k, :NB].mean(1); mse_eq = float(np.mean((y[k] - eq)**2))
        amb = float(np.mean(P[k, :NB].var(1)) / mse_eq)
        orc = float(np.sqrt(np.mean(np.min(e**2, 1)))); rmse_eq = float(np.sqrt(mse_eq))
        past = np.abs(D["E"][k]).mean(1); hit = float(np.mean(np.argmin(past, 1) == np.argmin(np.abs(e), 1)))
        # persistensi keunggulan: korelasi peringkat galat masa lalu vs galat saat ini, dirata-rata lintas waktu
        from scipy.stats import spearmanr
        rho = float(np.nanmean([spearmanr(past[t], np.abs(e[t]))[0] for t in range(len(e))]))
        dg[h] = {"korelasi_galat": corr, "ambiguitas_per_MSE": amb, "RMSE_oracle_per_equal": orc / rmse_eq, "hit_rate": hit,
                 "hit_acak": 1 / M, "rho_persistensi": rho}
    yv, okv = load(st); mon = (np.arange(N) % 12) + 1
    clim = np.array([yv[:VA0][okv[:VA0] & (mon[:VA0] == m)].mean() for m in range(1, 13)])
    an = yv[:VA0] - clim[mon[:VA0] - 1]; kk = okv[1:VA0] & okv[:VA0 - 1]
    r1 = np.corrcoef(an[:-1][kk], an[1:][kk])[0, 1]
    dg["R2_AR1_anomali"] = float(r1**2); dg["var_klimatologi"] = float(1 - np.var(an[okv[:VA0]]) / np.var(yv[:VA0][okv[:VA0]]))
    R["diag"][st] = dg

# Bobot GG4 per kelompok model dan per musim (musim basah = 6 bulan terbasah menurut klimatologi pra-2010)
FAM = {"Statistik": ["SARIMA"], "Machine learning": ["SVR", "XGBoost"], "Deep learning": ["LSTM", "GRU"],
       "Hibrida": ["XGBoost-VMD", "XGBoost-WT", "LSTM-VMD", "LSTM-WT"], "Klimatologi": ["Klim-ekspansif", "Klim-30th"]}
gfile = os.path.join(OUT, "pred_global_GG4.npz")
if os.path.exists(gfile):
    Z = np.load(gfile); idx, Wt = Z["idx"], Z["w"]
    for st in TESTSTS:
        si = sts.index(st); yv, okv = load(st); mon = (np.arange(N) % 12) + 1
        clim = np.array([yv[:VA0][okv[:VA0] & (mon[:VA0] == m)].mean() for m in range(1, 13)]); wet = set((np.argsort(clim)[-6:] + 1).tolist())
        R["bobot"][st] = {}
        for j, h in enumerate(HORIZONS):
            k = (D["st"][idx] == si) & (D["j"][idx] == j)
            Wf = np.column_stack([Wt[k][:, [pool.index(n) for n in FAM[f]]].sum(1) for f in FAM])
            mm = (D["T"][idx][k] % 12) + 1; isw = np.isin(mm, list(wet))
            R["bobot"][st][h] = {"semua": dict(zip(FAM, Wf.mean(0).round(3).tolist())), "basah": dict(zip(FAM, Wf[isw].mean(0).round(3).tolist())),
                                 "kering": dict(zip(FAM, Wf[~isw].mean(0).round(3).tolist()))}
json.dump(R, open(os.path.join(OUT, "hasil_evaluasi.json"), "w", newline="\n"), indent=1)

# Ringkasan cetak
for per in PER:
    print("\n== NSE rata-rata lintas stasiun uji,", per)
    for h in HORIZONS:
        common_st = [s for s in TESTSTS if h in R["nse"][per].get(s, {}) and all(m in R["nse"][per][s][h] for m in METHODS)]
        vals = {m: float(np.mean([R["nse"][per][s][h][m] for s in common_st])) for m in METHODS}
        R.setdefault("nse_rerata", {}).setdefault(per, {})[h] = {"stasiun": len(common_st), **vals}
        print(f"h={h} ({len(common_st)} stasiun)", {m: round(v, 3) for m, v in vals.items()})
json.dump(R, open(os.path.join(OUT, "hasil_evaluasi.json"), "w", newline="\n"), indent=1)
print("\n== Menang/kalah per stasiun-horizon (2015-2025)")
for k, v in R["menang"].items(): print(k, v)
print("\n== DM panel")
for k, v in R["panel"].items(): print(k, {kk: (round(x["DM"], 2), round(x["p"], 3), round(x["p_holm_horizon"], 3)) for kk, x in v.items() if kk.startswith("2015-2025")})
