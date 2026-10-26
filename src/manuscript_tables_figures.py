# Stage 6b: manuscript Tables 1-6 and Figures 1-4, plus the key figures quoted in the manuscript text.
#   python src/manuscript_tables_figures.py      -> writes reports/manuscript/table*.csv and figure*.png
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import OUT, DATA_PROCESSED, REPORTS, N, VA0

OUTN = os.path.join(REPORTS, "manuscript"); os.makedirs(OUTN, exist_ok=True)
H = ["1", "3", "6", "12"]
CODE = {"Clim-30yr": "Clim30", "Mean": "Mean", "NNLS": "NNLS", "L-G3": "L-G3", "GG3": "GG3", "GG4-transfer": "GG4-T"}
W = 3.2   # IJIES single-column width (inches)


def _R(): return json.load(open(os.path.join(OUT, "evaluation_results.json")))
def _MON(): return json.load(open(os.path.join(DATA_PROCESSED, "monthly_15.json")))
def _G(): return json.load(open(os.path.join(OUT, "error_metrics.json")))
def _lab(c): return " ".join({"Clim-30yr": "Clim30", "GG4-transfer": "GG4-T"}.get(x, x) for x in c.split(" "))


# ------------------------------------------------------------------ tables
def table1_stations():
    R = _R(); cov = pd.read_csv(os.path.join(DATA_PROCESSED, "observation_coverage.csv"), dtype={"id": str})
    n = {st: R["nse"]["2015-2025"][st]["1"]["n"] for st in R["stations"] if "1" in R["nse"]["2015-2025"].get(st, {})}
    return pd.DataFrame({"Station": cov["station"].str.replace("_", " "), "BoM ID": cov["id"], "Climate": cov["climate"],
                         "Latitude": cov["lat"], "Longitude": cov["lon"], "Rainfall (mm/month)": cov["mean_mm"],
                         "n test (h=1)": [n.get(s, np.nan) for s in cov["station"]]})


def table2_nse():
    R = _R(); rows = []
    for per, lab in [("2015-2025", "Test 2015-2025 (13 stations)"), ("2015-2020", "Test 2015-2020 (13 stations)"), ("2021-2025", "Test 2021-2025 (11 stations)")]:
        for h in H:
            d = R["nse_mean"][per][h]
            rows.append({"Period": lab, "h": int(h), **{CODE[m]: round(d[m], 3) for m in CODE}})
    return pd.DataFrame(rows)


PANEL_SET = [("2015-2025", ["GG3 vs L-G3", "GG4 vs L-G4", "GG3 vs NNLS", "GG3 vs Mean", "GG3 vs Clim-30yr", "L-G4 vs NNLS",
                            "GG4-transfer vs GG4", "GG4-transfer vs Clim-30yr"]),
             ("2021-2025", ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Clim-30yr", "GG4-transfer vs L-G4", "GG4-transfer vs NNLS"])]


def table3_dm_panel():
    R = _R(); rows = []
    for per, comps in PANEL_SET:
        for c in comps:
            r = {"Period": per, "Comparison": _lab(c)}
            for h in H:
                d = R["panel"][c][f"{per} h={h}"]; r[f"DM h={h}"] = round(d["DM"], 2); r[f"p Holm h={h}"] = round(d["p_holm_horizon"], 3)
            rows.append(r)
    return pd.DataFrame(rows)


def table4_errors():
    G = _G(); rows = []
    for met, lab, k in [("RMSE (mm)", "RMSE", 2), ("MAE (mm)", "MAE", 2), ("MAE skill vs Clim30 (%)", "skill_MAE", 1)]:
        for h in H:
            d = G["mean"]["2015-2025"][h]
            v = {CODE[m]: round(100 * d[m][lab], 1) if lab == "skill_MAE" else round(d[m][lab], 2) for m in CODE}
            if lab == "skill_MAE": v["Clim30"] = np.nan
            rows.append({"Measure": met, "h": int(h), **v})
    return pd.DataFrame(rows)


DM_ABS_SET = [("2015-2025", ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Mean", "GG3 vs Clim-30yr", "GG4-transfer vs Clim-30yr"]),
              ("2021-2025", ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Clim-30yr"])]


def table5_dm_abs():
    G = _G(); rows = []
    for per, comps in DM_ABS_SET:
        for c in comps:
            r = {"Period": per, "Comparison": _lab(c)}
            for h in H:
                d = G["panel"][c][f"{per} h={h}"]["absolute"]; r[f"DM h={h}"] = round(d["DM"], 2); r[f"p Holm h={h}"] = round(d["p_holm_horizon"], 3)
            rows.append(r)
    return pd.DataFrame(rows)


def table6_nse_stations():
    R = _R(); rows = []
    for st in R["stations"]:
        d = R["nse"]["2015-2025"].get(st, {})
        if not d: continue
        rows.append({"Station": st.replace("_", " "), "n test (h=1)": d["1"]["n"],
                     **{CODE[m]: round(float(np.mean([d[h][m] for h in d if m in d[h]])), 3) for m in CODE}})
    return pd.DataFrame(rows)


def all_tables(save=True):
    T = {"table1_stations": table1_stations(), "table2_nse_mean": table2_nse(), "table3_dm_panel": table3_dm_panel(),
         "table4_rmse_mae": table4_errors(), "table5_dm_absolute_error": table5_dm_abs(), "table6_nse_stations": table6_nse_stations()}
    if save:
        for k, v in T.items(): v.to_csv(os.path.join(OUTN, f"{k}.csv"), index=False)
    return T


# ------------------------------------------------------------------ key figures quoted in the manuscript text
def key_figures():
    R = _R(); P = R["panel"]; out = {}
    for c in ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Clim-30yr", "GG3 vs Mean", "GG4 vs L-G4", "GG4-transfer vs GG4", "GG4-transfer vs Clim-30yr"]:
        out[c] = {"wins (of 52)": R["wins"][c]["better"],
                  "median MSE reduction (%)": round(100 * float(np.median([r["dMSE"] for r in R["dm"][c]])), 1),
                  "p Holm panel 2015-2025 (h=1,3,6,12)": [round(P[c][f"2015-2025 h={h}"]["p_holm_horizon"], 3) for h in H],
                  "p Holm panel 2021-2025 (h=1,3,6,12)": [round(P[c][f"2021-2025 h={h}"]["p_holm_horizon"], 3) for h in H]}
    return pd.DataFrame(out).T


def error_figures():
    """RMSE/MAE summary: wins per station-horizon by MAE, median MAE reduction, and Holm p of the panel DM test for both loss functions."""
    G = _G(); out = {}
    for c in ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Clim-30yr", "GG3 vs Mean", "GG4-transfer vs Clim-30yr"]:
        mm = G["wins_mae"][c]["2015-2025"]
        out[_lab(c)] = {"MAE wins (of 52)": mm["wins"], "median MAE reduction (%)": round(100 * mm["median_MAE_reduction"], 1),
                        "p Holm squared error (h=1,3,6,12)": [round(G["panel"][c][f"2015-2025 h={h}"]["squared"]["p_holm_horizon"], 3) for h in H],
                        "p Holm absolute error (h=1,3,6,12)": [round(G["panel"][c][f"2015-2025 h={h}"]["absolute"]["p_holm_horizon"], 3) for h in H]}
    return pd.DataFrame(out).T


# ------------------------------------------------------------------ figures
def _style():
    from matplotlib import font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    font = next((f for f in ["Liberation Serif", "Times New Roman", "DejaVu Serif"] if f in available), "serif")   # Liberation Serif = Times New Roman metrics
    plt.rcdefaults()
    plt.rcParams.update({"font.family": font, "font.size": 7.5,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.frameon": False})


def figure1_framework(path=None):
    _style(); path = path or os.path.join(OUTN, "figure1_framework.png")
    fig, ax = plt.subplots(figsize=(W, 4.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 14.2); ax.axis("off")
    def box(x, y, w, h, text, fc="#f4f6f8", ec="#37474f", bold=False, fs=6.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.2)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#37474f"))
    box(0.3, 12.6, 9.4, 1.4, "SILO daily rainfall, 15 stations, 1926–2025\nmonthly aggregation + observation mask (≥ 90% of days)")
    arrow(5, 12.6, 5, 12.05)
    box(0.3, 10.15, 9.4, 1.85, "Base models, retrained every 5 years from 1961:\nSARIMA, SVR, XGBoost, LSTM, GRU, VMD/WT hybrids,\nexpanding and 30-year climatologies (11 members)")
    arrow(5, 10.15, 5, 9.6)
    box(0.3, 7.55, 9.4, 2.0, "Gate inputs per station–month–horizon:\n12-month error history × 11 models, model forecasts,\nmonth, ENSO/IOD regime, horizon\n(meta-training 1962–2009, validation 2010–2014)")
    for x in [1.75, 5.0, 8.25]: arrow(5, 7.55, x, 6.95)
    box(0.2, 5.0, 3.1, 1.9, "Local gate\n(L-G3, L-G4)\ntrained per\nstation", fc="#fff3e0", ec="#e65100")
    box(3.45, 5.0, 3.1, 1.9, "Global gate\n(GG3, GG4)\n15 stations +\nembedding", fc="#e3f2fd", ec="#0d47a1", bold=True)
    box(6.7, 5.0, 3.1, 1.9, "Transfer gate\n(GG4-T)\nwithout test-\nstation data", fc="#e8f5e9", ec="#1b5e20")
    for x in [1.75, 5.0, 8.25]: arrow(x, 5.0, 5, 4.35)
    box(0.3, 2.65, 9.4, 1.65, "Compared with static benchmarks:\nmean, NNLS, 30-year rolling climatology", fc="#fafafa")
    arrow(5, 2.65, 5, 2.1)
    box(0.3, 0.2, 9.4, 1.85, "Test 2015–2025 (sub-period 2021–2025): NSE,\nper-station DM tests (Holm correction),\ncross-station panel DM tests")
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02); return fig


def figure2_skill(path=None):
    _style(); R = _R(); path = path or os.path.join(OUTN, "figure2_skill.png")
    show = [("Mean", "Mean", "#9e9e9e", "s", 1.0), ("NNLS", "NNLS", "#424242", "D", 1.0), ("L-G3", "Local gate L-G3", "#e69f00", "v", 1.0),
            ("GG3", "Global gate GG3", "#0072b2", "o", 1.6), ("GG4-transfer", "Transfer gate GG4-T", "#009e73", "^", 1.3)]
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.1), sharex=True)
    for ax, per in zip(axes, ["2015-2020", "2021-2025"]):
        for key, lab, c, mk, lw in show:
            vals = []
            for h in H:
                sk = [(R["nse"][per][s][h][key] - R["nse"][per][s][h]["Clim-30yr"]) / (1 - R["nse"][per][s][h]["Clim-30yr"])
                      for s in R["stations"] if h in R["nse"][per].get(s, {}) and key in R["nse"][per][s][h] and "Clim-30yr" in R["nse"][per][s][h]]
                vals.append(100 * np.mean(sk))
            ax.plot([1, 3, 6, 12], vals, marker=mk, ms=3.2, lw=lw, color=c, label=lab)
        ax.axhline(0, color="k", lw=0.6, ls="--"); ax.set_ylabel("MSE skill (%)"); ax.set_title(f"Test {per.replace('-', '–')}", fontsize=7.5)
        ax.grid(alpha=0.25, lw=0.4)
    axes[1].set_xticks([1, 3, 6, 12]); axes[1].set_xlabel("Horizon (months)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(h_pad=0.6, rect=(0, 0, 1, 0.9))
    fig.legend(h_, l_, fontsize=6.3, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=0.9)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02); return fig


def figure3_heatmap(path=None):
    _style(); R = _R(); path = path or os.path.join(OUTN, "figure3_heatmap.png")
    keys = [("GG3 vs L-G3", "(a) GG3 vs L-G3"), ("GG3 vs NNLS", "(b) GG3 vs NNLS")]
    sts = [s for s in R["stations"] if any(r["st"] == s for r in R["dm"]["GG3 vs L-G3"])]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.45), sharey=True); mats = []
    for (k, _) in keys:
        Mx = np.full((len(sts), 4), np.nan); Sg = np.zeros((len(sts), 4), bool)
        for r in R["dm"][k]:
            i = sts.index(r["st"]); j = [1, 3, 6, 12].index(r["h"]); Mx[i, j] = 100 * r["dMSE"]; Sg[i, j] = r["p"] < 0.05
        mats.append((Mx, Sg))
    lim = max(np.nanmax(np.abs(m)) for m, _ in mats)
    for ax, (Mx, Sg), (_, ttl) in zip(axes, mats, keys):
        im = ax.imshow(Mx, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(len(sts)):
            for j in range(4):
                v = Mx[i, j]; col = "white" if abs(v) > 0.6 * lim else "black"
                ax.text(j, i, f"{v:.1f}{'*' if Sg[i, j] else ''}", ha="center", va="center", fontsize=5.6, color=col)
        ax.set_xticks(range(4)); ax.set_xticklabels(H); ax.set_xlabel("Horizon (months)"); ax.set_title(ttl, fontsize=7)
    axes[0].set_yticks(range(len(sts))); axes[0].set_yticklabels([s.replace("_", " ") for s in sts], fontsize=6.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02); cb.set_label("MSE reduction (%)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02); return fig


def figure4_weights(path=None):
    _style(); path = path or os.path.join(OUTN, "figure4_weights.png")
    Z = np.load(os.path.join(OUT, "meta_dataset.npz")); pool = list(Z["pool"]); stl = list(Z["stations"])
    G = np.load(os.path.join(OUT, "pred_global_GG3.npz")); idx, Wg = G["idx"], G["w"]; sti, jj, TT = Z["st"][idx], Z["j"][idx], Z["T"][idx]
    FAM = [("SARIMA", ["SARIMA"], "#cc79a7"), ("ML", ["SVR", "XGBoost"], "#e69f00"), ("DL", ["LSTM", "GRU"], "#56b4e9"),
           ("Hybrid", ["XGBoost-VMD", "XGBoost-WT", "LSTM-VMD", "LSTM-WT"], "#009e73"), ("Climatology", ["Clim-expanding", "Clim-30yr"], "#8c8c8c")]
    MON = _MON()
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.7), sharex=True)
    for ax, st in zip(axes, ["Darwin", "Perth"]):
        k = (sti == stl.index(st)) & (jj == 0); mm = (TT[k] % 12) + 1
        stack = np.array([[Wg[k][mm == m][:, [pool.index(n) for n in mem]].sum(1).mean() for m in range(1, 13)] for _, mem, _ in FAM])
        ax.stackplot(range(1, 13), stack, colors=[c for *_, c in FAM], labels=[f for f, *_ in FAM], lw=0)
        rain = np.array(MON["stations"][st]["rain"]); ok = np.array(MON["stations"][st]["ok"], bool); mo = (np.arange(N) % 12) + 1
        clim = np.array([rain[:VA0][ok[:VA0] & (mo[:VA0] == m)].mean() for m in range(1, 13)]); wet = set((np.argsort(clim)[-6:] + 1).tolist())
        for m in wet: ax.axvspan(m - 0.5, m + 0.5, ymin=0.97, ymax=1.0, color="#1565c0", lw=0)
        ax.set_ylim(0, 1); ax.set_xlim(1, 12); ax.set_ylabel("Weight"); ax.set_title(f"{st} (blue bar = 6 wettest months)", fontsize=7)
    axes[1].set_xticks(range(1, 13)); axes[1].set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]); axes[1].set_xlabel("Target month")
    axes[0].legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.42), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(h_pad=0.7); fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02); return fig


def all_figures():
    figs = [figure1_framework(), figure2_skill(), figure3_heatmap(), figure4_weights()]
    for f in figs: plt.close(f)


if __name__ == "__main__":
    for k, v in all_tables().items(): print("\n==", k); print(v.to_string(index=False))
    print("\n== key figures"); print(key_figures().to_string())
    print("\n== RMSE/MAE and absolute-error DM tests"); print(error_figures().to_string())
    all_figures(); print("\nFigures saved in", OUTN)
