# Stage 6a: tables and figures of the full experiment report (all comparisons, diagnostics, weights) in reports/experiment/.
#   python src/experiment_report.py
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from common import OUT, DATA_PROCESSED, REPORTS
OUTF = os.path.join(REPORTS, "experiment"); os.makedirs(OUTF, exist_ok=True)
R = json.load(open(os.path.join(OUT, "evaluation_results.json")))
H = ["1", "3", "6", "12"]; PER = ["2015-2025", "2015-2020", "2021-2025"]
MAIN = ["Clim-30yr", "Clim-expanding", "Mean", "NNLS", "Best model", "L-G3", "L-G4", "GG3", "GG4", "GG4-transfer"]
LAB = {"Clim-30yr": "Climatology 30 yr", "Clim-expanding": "Expanding climatology", "Mean": "Mean", "NNLS": "NNLS",
       "Best model": "Best model", "L-G3": "Local gate G3", "L-G4": "Local gate G4", "GG3": "Global gate GG3",
       "GG4": "Global gate GG4", "GG4-transfer": "Global transfer gate"}
T = {}
# Table 1: mean NSE across stations (stations complete for all methods)
rows = []
for per in PER:
    for h in H:
        r = R["nse_mean"][per][h]; rows.append({"Period": per, "h": int(h), "n stations": r["stations"], **{LAB[m]: round(r[m], 3) for m in MAIN}})
T["nse_mean"] = pd.DataFrame(rows)
# Table 2: NSE per station, mean of the 4 horizons, 2015-2025
rows = []
for st in R["stations"]:
    d = R["nse"]["2015-2025"].get(st, {})
    if not d: continue
    rows.append({"Station": st.replace("_", " "), "Test months (h=1)": d["1"]["n"], **{LAB[m]: round(float(np.mean([d[h][m] for h in d if m in d[h]])), 3) for m in MAIN}})
T["nse_stations"] = pd.DataFrame(rows)
# Table 3: panel DM with Holm across horizons
KEYS = ["GG3 vs L-G3", "GG4 vs L-G4", "GG3 vs NNLS", "GG3 vs Mean", "GG3 vs Clim-30yr", "GG4 vs NNLS", "GG4 vs Mean", "GG4 vs Clim-30yr",
        "GG3 vs GG4", "L-G4 vs NNLS", "L-G4 vs Clim-30yr", "NNLS vs Clim-30yr", "GG4-transfer vs GG4", "GG4-transfer vs L-G4", "GG4-transfer vs NNLS", "GG4-transfer vs Clim-30yr"]
rows = []
for k in KEYS:
    for per in PER:
        for h in H:
            x = R["panel"][k].get(f"{per} h={h}")
            if x: rows.append({"Comparison": k, "Period": per, "h": int(h), "DM": round(x["DM"], 2), "p": round(x["p"], 3), "p Holm (4 horizons)": round(x["p_holm_horizon"], 3), "months": x["months"]})
T["dm_panel"] = pd.DataFrame(rows)
# Table 4: wins/losses per station-horizon
T["wins"] = pd.DataFrame([{"Comparison": k, **{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}} for k, v in R["wins"].items()])
# Table 5: ensemble-suitability diagnostics + gain in the test period
dm = {k: {(r["st"], r["h"]): r["dMSE"] for r in R["dm"][k]} for k in R["dm"]}
rows = []
for st, dg in R["diag"].items():
    def avg(key): return float(np.mean([dg[h][key] for h in H]))
    g = lambda k: float(np.mean([dm[k][(st, int(h))] for h in H if (st, int(h)) in dm[k]])) if any((st, int(h)) in dm[k] for h in H) else np.nan
    rows.append({"Station": st.replace("_", " "), "error correlation": round(avg("error_correlation"), 3), "ambiguity/MSE": round(avg("ambiguity_per_MSE"), 3),
                 "RMSE oracle/mean": round(avg("RMSE_oracle_per_equal"), 3), "hit rate": round(avg("hit_rate"), 3), "rho persistence": round(avg("rho_persistence"), 3),
                 "R2 AR1 anomaly": round(dg["R2_AR1_anomaly"], 3), "variance explained by climatology": round(dg["var_climatology"], 3),
                 "gain L-G4 vs NNLS": round(g("L-G4 vs NNLS"), 4), "gain GG4 vs L-G4": round(g("GG4 vs L-G4"), 4), "gain GG4 vs Clim-30yr": round(g("GG4 vs Clim-30yr"), 4)})
T["diagnostics"] = pd.DataFrame(rows)
DIAGS = ["error correlation", "ambiguity/MSE", "RMSE oracle/mean", "hit rate", "rho persistence", "R2 AR1 anomaly", "variance explained by climatology"]
GAINS = ["gain L-G4 vs NNLS", "gain GG4 vs L-G4", "gain GG4 vs Clim-30yr"]
rows = []
for dname in DIAGS:
    r = {"Diagnostic": dname}
    for gname in GAINS:
        k = T["diagnostics"][[dname, gname]].dropna(); rho, p = spearmanr(k[dname], k[gname]); r[f"{gname}: rho"] = round(rho, 2); r[f"{gname}: p"] = round(p, 3)
    rows.append(r)
T["diagnostics_correlation"] = pd.DataFrame(rows)
# Table 6: GG4 weights per model family, wet vs dry season, h=1 and h=12
rows = []
for st, bb in R["weights"].items():
    for h in ["1", "12"]:
        for season in ["wet", "dry"]:
            rows.append({"Station": st.replace("_", " "), "h": int(h), "Season": season, **bb[h][season]})
T["weights"] = pd.DataFrame(rows)

# Figure 1: mean skill relative to the 30-year climatology per horizon (MSESS from NSE: (NSE_m - NSE_k)/(1 - NSE_k)), per station then averaged
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
show = ["Mean", "NNLS", "L-G4", "GG3", "GG4", "GG4-transfer"]
col = {"Mean": "#9e9e9e", "NNLS": "#616161", "L-G4": "#e69f00", "GG3": "#56b4e9", "GG4": "#0072b2", "GG4-transfer": "#009e73"}
for ax, per in zip(axes, ["2015-2020", "2021-2025"]):
    for m in show:
        vals = []
        for h in H:
            sk = [(R["nse"][per][s][h][m] - R["nse"][per][s][h]["Clim-30yr"]) / (1 - R["nse"][per][s][h]["Clim-30yr"])
                  for s in R["stations"] if h in R["nse"][per].get(s, {}) and m in R["nse"][per][s][h] and "Clim-30yr" in R["nse"][per][s][h]]
            vals.append(np.mean(sk))
        ax.plot([1, 3, 6, 12], vals, marker="o", label=LAB[m], color=col[m], lw=2 if m.startswith("GG") else 1.2)
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.set_xticks([1, 3, 6, 12]); ax.set_xlabel("Horizon (months)"); ax.set_title(f"Test {per}")
axes[0].set_ylabel("MSE skill vs 30-yr climatology\n(station mean)"); axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(-0.05, -0.18), ncol=6, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUTF, "figure1_skill_horizon.png"), dpi=200, bbox_inches="tight"); plt.close()
# Figures 2, 3, 6: heat maps of the MSE difference per station-horizon
for key, fn, ttl in [("GG3 vs L-G3", "figure2_global_vs_local.png", "MSE improvement of global gate GG3 over local gate L-G3"),
                     ("GG3 vs NNLS", "figure3_global_vs_nnls.png", "MSE improvement of global gate GG3 over NNLS"),
                     ("GG4-transfer vs Clim-30yr", "figure6_transfer_vs_clim.png", "MSE improvement of the transfer gate over the 30-yr climatology")]:
    sts = [s for s in R["stations"] if any(r["st"] == s for r in R["dm"][key])]
    M = np.full((len(sts), 4), np.nan); S = np.zeros((len(sts), 4), bool)
    for r in R["dm"][key]:
        i = sts.index(r["st"]); j = [1, 3, 6, 12].index(r["h"]); M[i, j] = 100 * r["dMSE"]; S[i, j] = r["p"] < 0.05
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(sts) + 1.2)); lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(sts)):
        for j in range(4):
            if np.isfinite(M[i, j]): ax.text(j, i, f"{M[i, j]:.1f}{'*' if S[i, j] else ''}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["h=1", "h=3", "h=6", "h=12"]); ax.set_yticks(range(len(sts))); ax.set_yticklabels([s.replace("_", " ") for s in sts])
    ax.set_title(ttl + "\n(% MSE reduction; * p < 0.05 before Holm)", fontsize=9); plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(OUTF, fn), dpi=200); plt.close()
# Figure 4: GG3 weights (global gate without anchor) per family, wet vs dry season, h=1; wet season = 6 wettest months before 2010
Z = np.load(os.path.join(OUT, "meta_dataset.npz")); pool = list(Z["pool"]); stl = list(Z["stations"]); G = np.load(os.path.join(OUT, "pred_global_GG3.npz"))
idx, Wg = G["idx"], G["w"]; sti, jj, TT = Z["st"][idx], Z["j"][idx], Z["T"][idx]
MON = json.load(open(os.path.join(DATA_PROCESSED, "monthly_15.json")))
FAMS = {"Statistical (SARIMA)": ["SARIMA"], "Machine learning": ["SVR", "XGBoost"], "Deep learning": ["LSTM", "GRU"],
        "Hybrid": ["XGBoost-VMD", "XGBoost-WT", "LSTM-VMD", "LSTM-WT"], "Climatology": ["Clim-expanding", "Clim-30yr"]}
FC = ["#cc79a7", "#e69f00", "#56b4e9", "#009e73", "#7f7f7f"]; rowsw = []
sts = [s for s in R["stations"] if s in stl]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, season in zip(axes, ["wet", "dry"]):
    bottom = np.zeros(len(sts))
    for (f, members), c in zip(FAMS.items(), FC):
        vals = []
        for s in sts:
            rain = np.array(MON["stations"][s]["rain"]); ok = np.array(MON["stations"][s]["ok"], bool); mon = (np.arange(1200) % 12) + 1
            clim = np.array([rain[:1008][ok[:1008] & (mon[:1008] == m)].mean() for m in range(1, 13)]); wet = np.argsort(clim)[-6:] + 1
            k = (sti == stl.index(s)) & (jj == 0); mm = (TT[k] % 12) + 1; sel = np.isin(mm, wet) if season == "wet" else ~np.isin(mm, wet)
            vals.append(float(Wg[k][sel][:, [pool.index(n) for n in members]].sum(1).mean()))
        for s, v in zip(sts, vals): rowsw.append({"Station": s, "Season": season, "Family": f, "Weight": round(v, 3)})
        ax.barh([s.replace("_", " ") for s in sts], vals, left=bottom, color=c, label=f); bottom += np.array(vals)
    ax.set_title(f"{season.capitalize()} season (h = 1)"); ax.set_xlim(0, 1); ax.set_xlabel("Mean weight of global gate GG3")
axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(-0.05, -0.13), ncol=5, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUTF, "figure4_weights_season.png"), dpi=200, bbox_inches="tight"); plt.close()
T["weights_GG3"] = pd.DataFrame(rowsw)
sd = {n: float(np.mean([np.load(os.path.join(OUT, f"pred_global_{n}.npz"))["w"][(sti == si) & (jj == 0)].std(0).mean() for si in range(len(stl))])) for n in ["GG3", "GG4"]}
json.dump(sd, open(os.path.join(OUTF, "weights_temporal_sd.json"), "w"))
# Figure 5: diagnostics vs gain
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (dname, gname) in zip(axes, [("rho persistence", "gain L-G4 vs NNLS"), ("ambiguity/MSE", "gain L-G4 vs NNLS"), ("hit rate", "gain GG4 vs L-G4")]):
    k = T["diagnostics"][["Station", dname, gname]].dropna(); ax.scatter(k[dname], 100 * k[gname], color="#0072b2")
    for _, r in k.iterrows(): ax.annotate(r["Station"], (r[dname], 100 * r[gname]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    rho, p = spearmanr(k[dname], k[gname]); ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel(dname); ax.set_ylabel(gname + " (%)"); ax.set_title(f"Spearman rho = {rho:.2f}, p = {p:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUTF, "figure5_diagnostics_vs_gain.png"), dpi=200); plt.close()
with pd.ExcelWriter(os.path.join(OUTF, "tables_global_gate.xlsx")) as xw:
    for k, v in T.items(): v.to_excel(xw, sheet_name=k[:30], index=False)
for k, v in T.items(): v.to_csv(os.path.join(OUTF, f"table_{k}.csv"), index=False)
for k in ["nse_mean", "wins", "diagnostics", "diagnostics_correlation"]:
    print("\n==", k); print(T[k].to_string(index=False))
print("\n== dm_panel 2015-2025"); print(T["dm_panel"][T["dm_panel"].Period == "2015-2025"].to_string(index=False))
