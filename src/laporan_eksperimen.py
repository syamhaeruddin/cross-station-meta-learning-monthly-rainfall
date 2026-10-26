# Tahap 6a: tabel dan gambar laporan eksperimen lengkap (semua perbandingan, diagnosis, bobot) di laporan/eksperimen/.
#   python src/laporan_eksperimen.py
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from common import OUT, DATA_OLAHAN, LAPORAN
OUTF = os.path.join(LAPORAN, "eksperimen"); os.makedirs(OUTF, exist_ok=True)
R = json.load(open(os.path.join(OUT, "hasil_evaluasi.json")))
H = ["1", "3", "6", "12"]; PER = ["2015-2025", "2015-2020", "2021-2025"]
MAIN = ["Klim-30th", "Klim-ekspansif", "Rata-rata", "NNLS", "Model terbaik", "L-G3", "L-G4", "GG3", "GG4", "GG4-transfer"]
LAB = {"Klim-30th": "Klimatologi 30 th", "Klim-ekspansif": "Klimatologi ekspansif", "Rata-rata": "Rata-rata", "NNLS": "NNLS",
       "Model terbaik": "Model terbaik", "L-G3": "Gate lokal G3", "L-G4": "Gate lokal G4", "GG3": "Gate global GG3",
       "GG4": "Gate global GG4", "GG4-transfer": "Gate global transfer"}
T = {}
# Tabel 1: NSE rata-rata lintas stasiun (stasiun yang lengkap untuk semua metode)
rows = []
for per in PER:
    for h in H:
        r = R["nse_rerata"][per][h]; rows.append({"Periode": per, "h": int(h), "n stasiun": r["stasiun"], **{LAB[m]: round(r[m], 3) for m in MAIN}})
T["nse_rerata"] = pd.DataFrame(rows)
# Tabel 2: NSE per stasiun, rata-rata 4 horizon, 2015-2025
rows = []
for st in R["stasiun"]:
    d = R["nse"]["2015-2025"].get(st, {})
    if not d: continue
    rows.append({"Stasiun": st.replace("_", " "), "Bulan uji (h=1)": d["1"]["n"], **{LAB[m]: round(float(np.mean([d[h][m] for h in d if m in d[h]])), 3) for m in MAIN}})
T["nse_stasiun"] = pd.DataFrame(rows)
# Tabel 3: DM panel dengan Holm lintas horizon
KEYS = ["GG3 vs L-G3", "GG4 vs L-G4", "GG3 vs NNLS", "GG3 vs Rata-rata", "GG3 vs Klim-30th", "GG4 vs NNLS", "GG4 vs Rata-rata", "GG4 vs Klim-30th",
        "GG3 vs GG4", "L-G4 vs NNLS", "L-G4 vs Klim-30th", "NNLS vs Klim-30th", "GG4-transfer vs GG4", "GG4-transfer vs L-G4", "GG4-transfer vs NNLS", "GG4-transfer vs Klim-30th"]
rows = []
for k in KEYS:
    for per in PER:
        for h in H:
            x = R["panel"][k].get(f"{per} h={h}")
            if x: rows.append({"Perbandingan": k, "Periode": per, "h": int(h), "DM": round(x["DM"], 2), "p": round(x["p"], 3), "p Holm (4 horizon)": round(x["p_holm_horizon"], 3), "bulan": x["bulan"]})
T["dm_panel"] = pd.DataFrame(rows)
# Tabel 4: menang/kalah per stasiun-horizon
T["menang"] = pd.DataFrame([{"Perbandingan": k, **{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}} for k, v in R["menang"].items()])
# Tabel 5: diagnosis kelayakan ensemble + gain pada periode uji
dm = {k: {(r["st"], r["h"]): r["dMSE"] for r in R["dm"][k]} for k in R["dm"]}
rows = []
for st, dg in R["diag"].items():
    def avg(key): return float(np.mean([dg[h][key] for h in H]))
    g = lambda k: float(np.mean([dm[k][(st, int(h))] for h in H if (st, int(h)) in dm[k]])) if any((st, int(h)) in dm[k] for h in H) else np.nan
    rows.append({"Stasiun": st.replace("_", " "), "korelasi galat": round(avg("korelasi_galat"), 3), "ambiguitas/MSE": round(avg("ambiguitas_per_MSE"), 3),
                 "RMSE oracle/rata-rata": round(avg("RMSE_oracle_per_equal"), 3), "hit rate": round(avg("hit_rate"), 3), "rho persistensi": round(avg("rho_persistensi"), 3),
                 "R2 AR1 anomali": round(dg["R2_AR1_anomali"], 3), "varians dijelaskan klimatologi": round(dg["var_klimatologi"], 3),
                 "gain L-G4 vs NNLS": round(g("L-G4 vs NNLS"), 4), "gain GG4 vs L-G4": round(g("GG4 vs L-G4"), 4), "gain GG4 vs Klim-30th": round(g("GG4 vs Klim-30th"), 4)})
T["diag"] = pd.DataFrame(rows)
DIAGS = ["korelasi galat", "ambiguitas/MSE", "RMSE oracle/rata-rata", "hit rate", "rho persistensi", "R2 AR1 anomali", "varians dijelaskan klimatologi"]
GAINS = ["gain L-G4 vs NNLS", "gain GG4 vs L-G4", "gain GG4 vs Klim-30th"]
rows = []
for dname in DIAGS:
    r = {"Diagnosis": dname}
    for gname in GAINS:
        k = T["diag"][[dname, gname]].dropna(); rho, p = spearmanr(k[dname], k[gname]); r[f"{gname}: rho"] = round(rho, 2); r[f"{gname}: p"] = round(p, 3)
    rows.append(r)
T["diag_korelasi"] = pd.DataFrame(rows)
# Tabel 6: bobot GG4 per kelompok model, musim basah vs kering, h=1 dan h=12
rows = []
for st, bb in R["bobot"].items():
    for h in ["1", "12"]:
        for mus in ["basah", "kering"]:
            rows.append({"Stasiun": st.replace("_", " "), "h": int(h), "Musim": mus, **bb[h][mus]})
T["bobot"] = pd.DataFrame(rows)

# Gambar 1: skill rata-rata terhadap klimatologi 30 tahun per horizon (MSESS dari NSE: (NSE_m - NSE_k)/(1 - NSE_k)), per stasiun lalu dirata-rata
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
show = ["Rata-rata", "NNLS", "L-G4", "GG3", "GG4", "GG4-transfer"]
col = {"Rata-rata": "#9e9e9e", "NNLS": "#616161", "L-G4": "#e69f00", "GG3": "#56b4e9", "GG4": "#0072b2", "GG4-transfer": "#009e73"}
for ax, per in zip(axes, ["2015-2020", "2021-2025"]):
    for m in show:
        vals = []
        for h in H:
            sk = [(R["nse"][per][s][h][m] - R["nse"][per][s][h]["Klim-30th"]) / (1 - R["nse"][per][s][h]["Klim-30th"])
                  for s in R["stasiun"] if h in R["nse"][per].get(s, {}) and m in R["nse"][per][s][h] and "Klim-30th" in R["nse"][per][s][h]]
            vals.append(np.mean(sk))
        ax.plot([1, 3, 6, 12], vals, marker="o", label=LAB[m], color=col[m], lw=2 if m.startswith("GG") else 1.2)
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.set_xticks([1, 3, 6, 12]); ax.set_xlabel("Horizon (bulan)"); ax.set_title(f"Uji {per}")
axes[0].set_ylabel("Skill MSE vs klimatologi 30 th\n(rata-rata stasiun)"); axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(-0.05, -0.18), ncol=6, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUTF, "gambar1_skill_horizon.png"), dpi=200, bbox_inches="tight"); plt.close()
# Gambar 2: peta panas selisih MSE gate global vs gate lokal per stasiun-horizon
for key, fn, ttl in [("GG3 vs L-G3", "gambar2_global_vs_lokal.png", "Perbaikan MSE gate global GG3 terhadap gate lokal L-G3"),
                     ("GG3 vs NNLS", "gambar3_global_vs_nnls.png", "Perbaikan MSE gate global GG3 terhadap NNLS"),
                     ("GG4-transfer vs Klim-30th", "gambar6_transfer_vs_klim.png", "Perbaikan MSE gate transfer terhadap klimatologi 30 th")]:
    sts = [s for s in R["stasiun"] if any(r["st"] == s for r in R["dm"][key])]
    M = np.full((len(sts), 4), np.nan); S = np.zeros((len(sts), 4), bool)
    for r in R["dm"][key]:
        i = sts.index(r["st"]); j = [1, 3, 6, 12].index(r["h"]); M[i, j] = 100 * r["dMSE"]; S[i, j] = r["p"] < 0.05
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(sts) + 1.2)); lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(sts)):
        for j in range(4):
            if np.isfinite(M[i, j]): ax.text(j, i, f"{M[i, j]:.1f}{'*' if S[i, j] else ''}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["h=1", "h=3", "h=6", "h=12"]); ax.set_yticks(range(len(sts))); ax.set_yticklabels([s.replace("_", " ") for s in sts])
    ax.set_title(ttl + "\n(% penurunan MSE; * p < 0,05 sebelum Holm)", fontsize=9); plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(OUTF, fn), dpi=200); plt.close()
# Gambar 4: bobot GG3 (gate global tanpa jangkar) per kelompok, musim basah vs kering, h=1; musim basah = 6 bulan terbasah pra-2010
Z = np.load(os.path.join(OUT, "meta_dataset.npz")); pool = list(Z["pool"]); stl = list(Z["stations"]); G = np.load(os.path.join(OUT, "pred_global_GG3.npz"))
idx, Wg = G["idx"], G["w"]; sti, jj, TT = Z["st"][idx], Z["j"][idx], Z["T"][idx]
MON = json.load(open(os.path.join(DATA_OLAHAN, "monthly_15.json")))
FAMS = {"Statistik (SARIMA)": ["SARIMA"], "Machine learning": ["SVR", "XGBoost"], "Deep learning": ["LSTM", "GRU"],
        "Hibrida": ["XGBoost-VMD", "XGBoost-WT", "LSTM-VMD", "LSTM-WT"], "Klimatologi": ["Klim-ekspansif", "Klim-30th"]}
FC = ["#cc79a7", "#e69f00", "#56b4e9", "#009e73", "#7f7f7f"]; rowsw = []
sts = [s for s in R["stasiun"] if s in stl]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, mus in zip(axes, ["basah", "kering"]):
    bottom = np.zeros(len(sts))
    for (f, members), c in zip(FAMS.items(), FC):
        vals = []
        for s in sts:
            rain = np.array(MON["stations"][s]["rain"]); ok = np.array(MON["stations"][s]["ok"], bool); mon = (np.arange(1200) % 12) + 1
            clim = np.array([rain[:1008][ok[:1008] & (mon[:1008] == m)].mean() for m in range(1, 13)]); wet = np.argsort(clim)[-6:] + 1
            k = (sti == stl.index(s)) & (jj == 0); mm = (TT[k] % 12) + 1; sel = np.isin(mm, wet) if mus == "basah" else ~np.isin(mm, wet)
            vals.append(float(Wg[k][sel][:, [pool.index(n) for n in members]].sum(1).mean()))
        for s, v in zip(sts, vals): rowsw.append({"Stasiun": s, "Musim": mus, "Kelompok": f, "Bobot": round(v, 3)})
        ax.barh([s.replace("_", " ") for s in sts], vals, left=bottom, color=c, label=f); bottom += np.array(vals)
    ax.set_title(f"Musim {mus} (h = 1)"); ax.set_xlim(0, 1); ax.set_xlabel("Bobot rata-rata gate global GG3")
axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(-0.05, -0.13), ncol=5, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUTF, "gambar4_bobot_musim.png"), dpi=200, bbox_inches="tight"); plt.close()
T["bobot_GG3"] = pd.DataFrame(rowsw)
sd = {n: float(np.mean([np.load(os.path.join(OUT, f"pred_global_{n}.npz"))["w"][(sti == si) & (jj == 0)].std(0).mean() for si in range(len(stl))])) for n in ["GG3", "GG4"]}
json.dump(sd, open(os.path.join(OUTF, "sd_temporal_bobot.json"), "w"))
# Gambar 5: diagnosis vs gain
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (dname, gname) in zip(axes, [("rho persistensi", "gain L-G4 vs NNLS"), ("ambiguitas/MSE", "gain L-G4 vs NNLS"), ("hit rate", "gain GG4 vs L-G4")]):
    k = T["diag"][["Stasiun", dname, gname]].dropna(); ax.scatter(k[dname], 100 * k[gname], color="#0072b2")
    for _, r in k.iterrows(): ax.annotate(r["Stasiun"], (r[dname], 100 * r[gname]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    rho, p = spearmanr(k[dname], k[gname]); ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel(dname); ax.set_ylabel(gname + " (%)"); ax.set_title(f"Spearman rho = {rho:.2f}, p = {p:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUTF, "gambar5_diagnosis_vs_gain.png"), dpi=200); plt.close()
with pd.ExcelWriter(os.path.join(OUTF, "tabel_gate_global.xlsx")) as xw:
    for k, v in T.items(): v.to_excel(xw, sheet_name=k[:30], index=False)
for k, v in T.items(): v.to_csv(os.path.join(OUTF, f"tabel_{k}.csv"), index=False)
for k in ["nse_rerata", "menang", "diag", "diag_korelasi"]:
    print("\n==", k); print(T[k].to_string(index=False))
print("\n== dm_panel 2015-2025"); print(T["dm_panel"][T["dm_panel"].Periode == "2015-2025"].to_string(index=False))
