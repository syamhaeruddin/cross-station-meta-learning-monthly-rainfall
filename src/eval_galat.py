# Tahap 5b: RMSE, MAE, skill MAE, dan uji Diebold-Mariano panel dengan galat kuadrat dan galat absolut.
# Menulis hasil/hasil_galat.json. Dijalankan setelah src/eval_global.py (memakai dataset meta dan prediksi gate yang sama).
#   python src/eval_galat.py
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from meta_global import load_ds, station_nnls_prior

D = load_ds(); LP, W = station_nnls_prior(D)
sts = list(D["stations"]); pool = list(D["pool"]); NB = len(pool) - 2
P, y, SC = D["P"], D["y"], D["SC"]

# prakiraan setiap metode (sama dengan src/eval_global.py)
PRED = {"Klim-30th": P[:, pool.index("Klim-30th")], "Klim-ekspansif": P[:, pool.index("Klim-ekspansif")], "Rata-rata": P[:, :NB].mean(1),
        "NNLS": np.array([P[k] @ W[(D["st"][k], D["j"][k])] for k in range(len(y))])}
best = np.zeros(len(y))
for si in range(len(sts)):
    for j in range(4):
        k = (D["st"] == si) & (D["j"] == j); kp = k & (D["split"] < 2) & D["ok"]
        best[k] = P[k, int(np.argmin(((P[kp] - y[kp, None])**2).mean(0)))]
PRED["Model terbaik"] = best
def put(name, idx, arr):
    full = PRED.setdefault(name, np.full(len(y), np.nan)); full[idx] = arr.mean(0)
for f in glob.glob(os.path.join(OUT, "pred_local_*.npz")):
    Z = np.load(f)
    for nm in ["L-G3", "L-G4"]: put(nm, Z["idx"], Z[nm])
for nm in ["GG3", "GG4"]:
    Z = np.load(os.path.join(OUT, f"pred_global_{nm}.npz")); put(nm, Z["idx"], Z["pred"])
for f in glob.glob(os.path.join(OUT, "pred_transfer_*.npz")):
    Z = np.load(f); put("GG4-transfer", Z["idx"], Z["pred"])
METHODS = ["Klim-30th", "Klim-ekspansif", "Rata-rata", "NNLS", "Model terbaik", "L-G3", "L-G4", "GG3", "GG4", "GG4-transfer"]
TESTSTS = [s for s in sts if s not in ("Coffs_Harbour", "Melbourne")]
PER = {"2015-2025": (TE0, N), "2015-2020": (TE0, 1140), "2021-2025": (1140, N)}
def mask(si, j, per):
    a, b = PER[per]; return (D["st"] == si) & (D["j"] == j) & (D["split"] == 2) & D["ok"] & (D["T"] >= a) & (D["T"] < b)

# RMSE dan MAE (mm) per periode, horizon, stasiun, dan metode
rows = []
for per in PER:
    for j, h in enumerate(HORIZONS):
        for st in TESTSTS:
            k = mask(sts.index(st), j, per)
            if k.sum() < 12: continue
            for m in METHODS:
                e = (y[k] - PRED[m][k]) * SC[k]
                rows.append(dict(periode=per, h=h, stasiun=st, n=int(k.sum()), metode=m,
                                 RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e)))))
df = pd.DataFrame(rows)
R = {"per_stasiun": rows, "rerata": {}, "menang_mae": {}, "panel": {}}
for per in PER:
    d = df[df.periode == per]
    w = d.pivot_table(index=["h", "stasiun"], columns="metode", values="MAE")
    skill = 1 - w[METHODS].div(w["Klim-30th"], axis=0)          # skill MAE per stasiun, lalu dirata-rata
    R["rerata"][per] = {}
    for h in HORIZONS:
        dh = d[d.h == h]
        R["rerata"][per][h] = {m: {"RMSE": float(dh[dh.metode == m].RMSE.mean()), "MAE": float(dh[dh.metode == m].MAE.mean()),
                                   "skill_MAE": float(skill.loc[h, m].mean()), "stasiun": int((dh.metode == m).sum())} for m in METHODS}

PAIRS = [("GG3", "L-G3"), ("GG3", "NNLS"), ("GG3", "Rata-rata"), ("GG3", "Klim-30th"), ("GG4", "L-G4"),
         ("GG4-transfer", "Klim-30th"), ("GG4-transfer", "GG4"), ("GG4-transfer", "L-G4"), ("GG4-transfer", "NNLS")]
for a, b in PAIRS:
    key = f"{a} vs {b}"; R["menang_mae"][key] = {}; R["panel"][key] = {}
    for per in PER:
        w = df[df.periode == per].pivot_table(index=["h", "stasiun"], columns="metode", values="MAE")
        R["menang_mae"][key][per] = {"menang": int((w[a] < w[b]).sum()), "total": int(len(w)),
                                     "median_penurunan_MAE": float(np.median(1 - w[a] / w[b]))}
        for loss in ["kuadrat", "absolut"]:
            res = []
            for j, h in enumerate(HORIZONS):
                dd = {}
                for st in TESTSTS:
                    k = np.where(mask(sts.index(st), j, per))[0]
                    if len(k) < 12: continue
                    for kk in k:
                        ea, eb = y[kk] - PRED[a][kk], y[kk] - PRED[b][kk]   # galat terskala simpangan baku stasiun
                        dd.setdefault(int(D["T"][kk]), []).append(abs(ea) - abs(eb) if loss == "absolut" else ea**2 - eb**2)
                dser = np.array([np.mean(dd[t]) for t in sorted(dd)]); n = len(dser); dc = dser - dser.mean()
                gam = [np.sum(dc[l:] * dc[:n - l]) / n for l in range(h)]; v = gam[0] + 2 * sum(gam[1:]); v = v if v > 0 else gam[0]
                stat = dser.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
                res.append((h, float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1))), n))
            for (h, s_, p_, n), q in zip(res, holm([r[2] for r in res])):   # Holm atas 4 horizon
                R["panel"][key].setdefault(f"{per} h={h}", {})[loss] = {"DM": s_, "p": p_, "p_holm_horizon": float(q), "bulan": n}

with open(os.path.join(OUT, "hasil_galat.json"), "w", newline="\n") as f:
    json.dump(R, f, indent=1)

print("== RMSE dan MAE rata-rata lintas stasiun (mm), 2015-2025")
for h in HORIZONS:
    r = R["rerata"]["2015-2025"][h]
    print(f"h={h:2d}  RMSE:", {m: round(r[m]["RMSE"], 2) for m in ["Klim-30th", "Rata-rata", "NNLS", "L-G3", "GG3", "GG4-transfer"]})
    print(f"      MAE :", {m: round(r[m]["MAE"], 2) for m in ["Klim-30th", "Rata-rata", "NNLS", "L-G3", "GG3", "GG4-transfer"]})
print("\n== DM panel 2015-2025, p Holm (galat kuadrat | galat absolut), h = 1, 3, 6, 12")
for key in R["panel"]:
    print(key, [round(R["panel"][key][f"2015-2025 h={h}"]["kuadrat"]["p_holm_horizon"], 3) for h in HORIZONS], "|",
          [round(R["panel"][key][f"2015-2025 h={h}"]["absolut"]["p_holm_horizon"], 3) for h in HORIZONS])
