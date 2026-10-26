# Tahap 6b: Tabel 1-6 dan Gambar 1-4 naskah, serta angka-angka kunci yang dikutip di teks naskah.
#   python src/tabel_gambar_naskah.py      -> menulis laporan/naskah/tabel*.csv dan gambar*.png
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import OUT, DATA_OLAHAN, LAPORAN, N, VA0

OUTN = os.path.join(LAPORAN, "naskah"); os.makedirs(OUTN, exist_ok=True)
H = ["1", "3", "6", "12"]
KODE = {"Klim-30th": "Klim30", "Rata-rata": "Rata2", "NNLS": "NNLS", "L-G3": "L-G3", "GG3": "GG3", "GG4-transfer": "GG4-T"}
W = 3.2   # lebar satu kolom IJIES (inci)


def _R(): return json.load(open(os.path.join(OUT, "hasil_evaluasi.json")))
def _MON(): return json.load(open(os.path.join(DATA_OLAHAN, "monthly_15.json")))
def _G(): return json.load(open(os.path.join(OUT, "hasil_galat.json")))
def _lab(c): return " ".join({"Klim-30th": "Klim30", "Rata-rata": "Rata2", "GG4-transfer": "GG4-T"}.get(x, x) for x in c.split(" "))


# ------------------------------------------------------------------ tabel
def tabel1_stasiun():
    R = _R(); cov = pd.read_csv(os.path.join(DATA_OLAHAN, "cakupan_observasi.csv"), dtype={"id": str})
    n = {st: R["nse"]["2015-2025"][st]["1"]["n"] for st in R["stasiun"] if "1" in R["nse"]["2015-2025"].get(st, {})}
    return pd.DataFrame({"Stasiun": cov["stasiun"].str.replace("_", " "), "ID BoM": cov["id"], "Iklim": cov["iklim"],
                         "Lintang": cov["lat"], "Bujur": cov["lon"], "Hujan (mm/bln)": cov["mean_mm"],
                         "n uji (h=1)": [n.get(s, np.nan) for s in cov["stasiun"]]})


def tabel2_nse():
    R = _R(); rows = []
    for per, lab in [("2015-2025", "Uji 2015-2025 (13 stasiun)"), ("2015-2020", "Uji 2015-2020 (13 stasiun)"), ("2021-2025", "Uji 2021-2025 (11 stasiun)")]:
        for h in H:
            d = R["nse_rerata"][per][h]
            rows.append({"Periode": lab, "h": int(h), **{KODE[m]: round(d[m], 3) for m in KODE}})
    return pd.DataFrame(rows)


PANEL_SET = [("2015-2025", ["GG3 vs L-G3", "GG4 vs L-G4", "GG3 vs NNLS", "GG3 vs Rata-rata", "GG3 vs Klim-30th", "L-G4 vs NNLS",
                            "GG4-transfer vs GG4", "GG4-transfer vs Klim-30th"]),
             ("2021-2025", ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Klim-30th", "GG4-transfer vs L-G4", "GG4-transfer vs NNLS"])]


def tabel3_dm_panel():
    R = _R(); rows = []
    lab = lambda c: " ".join({"Klim-30th": "Klim30", "Rata-rata": "Rata2", "GG4-transfer": "GG4-T"}.get(x, x) for x in c.split(" "))
    for per, comps in PANEL_SET:
        for c in comps:
            r = {"Periode": per, "Perbandingan": lab(c)}
            for h in H:
                d = R["panel"][c][f"{per} h={h}"]; r[f"DM h={h}"] = round(d["DM"], 2); r[f"p Holm h={h}"] = round(d["p_holm_horizon"], 3)
            rows.append(r)
    return pd.DataFrame(rows)


def tabel4_galat():
    G = _G(); rows = []
    for met, lab, k in [("RMSE (mm)", "RMSE", 2), ("MAE (mm)", "MAE", 2), ("Skill MAE terhadap Klim30 (%)", "skill_MAE", 1)]:
        for h in H:
            d = G["rerata"]["2015-2025"][h]
            v = {KODE[m]: round(100 * d[m][lab], 1) if lab == "skill_MAE" else round(d[m][lab], 2) for m in KODE}
            if lab == "skill_MAE": v["Klim30"] = np.nan
            rows.append({"Ukuran": met, "h": int(h), **v})
    return pd.DataFrame(rows)


DM_ABS_SET = [("2015-2025", ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Rata-rata", "GG3 vs Klim-30th", "GG4-transfer vs Klim-30th"]),
              ("2021-2025", ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Klim-30th"])]


def tabel5_dm_abs():
    G = _G(); rows = []
    for per, comps in DM_ABS_SET:
        for c in comps:
            r = {"Periode": per, "Perbandingan": _lab(c)}
            for h in H:
                d = G["panel"][c][f"{per} h={h}"]["absolut"]; r[f"DM h={h}"] = round(d["DM"], 2); r[f"p Holm h={h}"] = round(d["p_holm_horizon"], 3)
            rows.append(r)
    return pd.DataFrame(rows)


def tabel6_nse_stasiun():
    R = _R(); rows = []
    for st in R["stasiun"]:
        d = R["nse"]["2015-2025"].get(st, {})
        if not d: continue
        rows.append({"Stasiun": st.replace("_", " "), "n uji (h=1)": d["1"]["n"],
                     **{KODE[m]: round(float(np.mean([d[h][m] for h in d if m in d[h]])), 3) for m in KODE}})
    return pd.DataFrame(rows)


def semua_tabel(simpan=True):
    T = {"tabel1_stasiun": tabel1_stasiun(), "tabel2_nse_rerata": tabel2_nse(), "tabel3_dm_panel": tabel3_dm_panel(),
         "tabel4_rmse_mae": tabel4_galat(), "tabel5_dm_galat_absolut": tabel5_dm_abs(), "tabel6_nse_stasiun": tabel6_nse_stasiun()}
    if simpan:
        for k, v in T.items(): v.to_csv(os.path.join(OUTN, f"{k}.csv"), index=False)
    return T


# ------------------------------------------------------------------ angka kunci di teks naskah
def angka_kunci():
    R = _R(); P = R["panel"]; out = {}
    for c in ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Klim-30th", "GG3 vs Rata-rata", "GG4 vs L-G4", "GG4-transfer vs GG4", "GG4-transfer vs Klim-30th"]:
        out[c] = {"menang (dari 52)": R["menang"][c]["lebih baik"],
                  "median penurunan MSE (%)": round(100 * float(np.median([r["dMSE"] for r in R["dm"][c]])), 1),
                  "p Holm panel 2015-2025 (h=1,3,6,12)": [round(P[c][f"2015-2025 h={h}"]["p_holm_horizon"], 3) for h in H],
                  "p Holm panel 2021-2025 (h=1,3,6,12)": [round(P[c][f"2021-2025 h={h}"]["p_holm_horizon"], 3) for h in H]}
    return pd.DataFrame(out).T


def angka_galat():
    """Ringkasan RMSE/MAE: menang per stasiun-horizon menurut MAE, median penurunan MAE, dan p Holm uji DM panel untuk kedua fungsi galat."""
    G = _G(); out = {}
    for c in ["GG3 vs L-G3", "GG3 vs NNLS", "GG3 vs Klim-30th", "GG3 vs Rata-rata", "GG4-transfer vs Klim-30th"]:
        mm = G["menang_mae"][c]["2015-2025"]
        out[_lab(c)] = {"menang MAE (dari 52)": mm["menang"], "median penurunan MAE (%)": round(100 * mm["median_penurunan_MAE"], 1),
                        "p Holm galat kuadrat (h=1,3,6,12)": [round(G["panel"][c][f"2015-2025 h={h}"]["kuadrat"]["p_holm_horizon"], 3) for h in H],
                        "p Holm galat absolut (h=1,3,6,12)": [round(G["panel"][c][f"2015-2025 h={h}"]["absolut"]["p_holm_horizon"], 3) for h in H]}
    return pd.DataFrame(out).T


# ------------------------------------------------------------------ gambar
def _gaya():
    from matplotlib import font_manager as fm
    ada = {f.name for f in fm.fontManager.ttflist}
    huruf = next((f for f in ["Liberation Serif", "Times New Roman", "DejaVu Serif"] if f in ada), "serif")   # Liberation Serif = metrik Times New Roman
    plt.rcdefaults()
    plt.rcParams.update({"font.family": huruf, "font.size": 7.5,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.frameon": False})


def gambar1_kerangka(path=None):
    _gaya(); path = path or os.path.join(OUTN, "gambar1_kerangka.png")
    fig, ax = plt.subplots(figsize=(W, 4.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 14.2); ax.axis("off")
    def box(x, y, w, h, text, fc="#f4f6f8", ec="#37474f", bold=False, fs=6.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.2)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#37474f"))
    box(0.3, 12.6, 9.4, 1.4, "Curah hujan harian SILO, 15 stasiun, 1926–2025\nagregasi bulanan + masker pengamatan (≥ 90% hari)")
    arrow(5, 12.6, 5, 12.05)
    box(0.3, 10.15, 9.4, 1.85, "Model dasar, dilatih ulang tiap 5 tahun sejak 1961:\nSARIMA, SVR, XGBoost, LSTM, GRU, hibrida VMD/WT,\nklimatologi ekspansif dan 30 tahun (11 anggota)")
    arrow(5, 10.15, 5, 9.6)
    box(0.3, 7.55, 9.4, 2.0, "Masukan gate per stasiun–bulan–horizon:\nriwayat galat 12 bulan × 11 model, prakiraan model,\nbulan, rezim ENSO/IOD, horizon\n(meta-latih 1962–2009, validasi 2010–2014)")
    for x in [1.75, 5.0, 8.25]: arrow(5, 7.55, x, 6.95)
    box(0.2, 5.0, 3.1, 1.9, "Gate lokal\n(L-G3, L-G4)\ndilatih per\nstasiun", fc="#fff3e0", ec="#e65100")
    box(3.45, 5.0, 3.1, 1.9, "Gate global\n(GG3, GG4)\n15 stasiun +\nembedding", fc="#e3f2fd", ec="#0d47a1", bold=True)
    box(6.7, 5.0, 3.1, 1.9, "Gate transfer\n(GG4-T)\ntanpa data\nstasiun uji", fc="#e8f5e9", ec="#1b5e20")
    for x in [1.75, 5.0, 8.25]: arrow(x, 5.0, 5, 4.35)
    box(0.3, 2.65, 9.4, 1.65, "Dibandingkan dengan pembanding statis:\nrata-rata, NNLS, klimatologi bergulir 30 tahun", fc="#fafafa")
    arrow(5, 2.65, 5, 2.1)
    box(0.3, 0.2, 9.4, 1.85, "Uji 2015–2025 (sub-periode 2021–2025): NSE,\nuji DM per stasiun (koreksi Holm),\nuji DM panel lintas stasiun")
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02); return fig


def gambar2_skill(path=None):
    _gaya(); R = _R(); path = path or os.path.join(OUTN, "gambar2_skill.png")
    show = [("Rata-rata", "Rata-rata", "#9e9e9e", "s", 1.0), ("NNLS", "NNLS", "#424242", "D", 1.0), ("L-G3", "Gate lokal L-G3", "#e69f00", "v", 1.0),
            ("GG3", "Gate global GG3", "#0072b2", "o", 1.6), ("GG4-transfer", "Gate transfer GG4-T", "#009e73", "^", 1.3)]
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.1), sharex=True)
    for ax, per in zip(axes, ["2015-2020", "2021-2025"]):
        for key, lab, c, mk, lw in show:
            vals = []
            for h in H:
                sk = [(R["nse"][per][s][h][key] - R["nse"][per][s][h]["Klim-30th"]) / (1 - R["nse"][per][s][h]["Klim-30th"])
                      for s in R["stasiun"] if h in R["nse"][per].get(s, {}) and key in R["nse"][per][s][h] and "Klim-30th" in R["nse"][per][s][h]]
                vals.append(100 * np.mean(sk))
            ax.plot([1, 3, 6, 12], vals, marker=mk, ms=3.2, lw=lw, color=c, label=lab)
        ax.axhline(0, color="k", lw=0.6, ls="--"); ax.set_ylabel("Skill MSE (%)"); ax.set_title(f"Uji {per.replace('-', '–')}", fontsize=7.5)
        ax.grid(alpha=0.25, lw=0.4)
    axes[1].set_xticks([1, 3, 6, 12]); axes[1].set_xlabel("Horizon (bulan)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(h_pad=0.6, rect=(0, 0, 1, 0.9))
    fig.legend(h_, l_, fontsize=6.3, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=0.9)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02); return fig


def gambar3_heatmap(path=None):
    _gaya(); R = _R(); path = path or os.path.join(OUTN, "gambar3_heatmap.png")
    keys = [("GG3 vs L-G3", "(a) GG3 terhadap L-G3"), ("GG3 vs NNLS", "(b) GG3 terhadap NNLS")]
    sts = [s for s in R["stasiun"] if any(r["st"] == s for r in R["dm"]["GG3 vs L-G3"])]
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
                ax.text(j, i, f"{v:.1f}{'*' if Sg[i, j] else ''}".replace(".", ","), ha="center", va="center", fontsize=5.6, color=col)
        ax.set_xticks(range(4)); ax.set_xticklabels(H); ax.set_xlabel("Horizon (bulan)"); ax.set_title(ttl, fontsize=7)
    axes[0].set_yticks(range(len(sts))); axes[0].set_yticklabels([s.replace("_", " ") for s in sts], fontsize=6.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02); cb.set_label("Penurunan MSE (%)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02); return fig


def gambar4_bobot(path=None):
    _gaya(); path = path or os.path.join(OUTN, "gambar4_bobot.png")
    Z = np.load(os.path.join(OUT, "meta_dataset.npz")); pool = list(Z["pool"]); stl = list(Z["stations"])
    G = np.load(os.path.join(OUT, "pred_global_GG3.npz")); idx, Wg = G["idx"], G["w"]; sti, jj, TT = Z["st"][idx], Z["j"][idx], Z["T"][idx]
    FAM = [("SARIMA", ["SARIMA"], "#cc79a7"), ("ML", ["SVR", "XGBoost"], "#e69f00"), ("DL", ["LSTM", "GRU"], "#56b4e9"),
           ("Hibrida", ["XGBoost-VMD", "XGBoost-WT", "LSTM-VMD", "LSTM-WT"], "#009e73"), ("Klimatologi", ["Klim-ekspansif", "Klim-30th"], "#8c8c8c")]
    MON = _MON()
    fig, axes = plt.subplots(2, 1, figsize=(W, 3.7), sharex=True)
    for ax, st in zip(axes, ["Darwin", "Perth"]):
        k = (sti == stl.index(st)) & (jj == 0); mm = (TT[k] % 12) + 1
        stack = np.array([[Wg[k][mm == m][:, [pool.index(n) for n in mem]].sum(1).mean() for m in range(1, 13)] for _, mem, _ in FAM])
        ax.stackplot(range(1, 13), stack, colors=[c for *_, c in FAM], labels=[f for f, *_ in FAM], lw=0)
        rain = np.array(MON["stations"][st]["rain"]); ok = np.array(MON["stations"][st]["ok"], bool); mo = (np.arange(N) % 12) + 1
        clim = np.array([rain[:VA0][ok[:VA0] & (mo[:VA0] == m)].mean() for m in range(1, 13)]); wet = set((np.argsort(clim)[-6:] + 1).tolist())
        for m in wet: ax.axvspan(m - 0.5, m + 0.5, ymin=0.97, ymax=1.0, color="#1565c0", lw=0)
        ax.set_ylim(0, 1); ax.set_xlim(1, 12); ax.set_ylabel("Bobot"); ax.set_title(f"{st} (garis biru = 6 bulan terbasah)", fontsize=7)
    axes[1].set_xticks(range(1, 13)); axes[1].set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]); axes[1].set_xlabel("Bulan target")
    axes[0].legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.42), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(h_pad=0.7); fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02); return fig


def semua_gambar():
    figs = [gambar1_kerangka(), gambar2_skill(), gambar3_heatmap(), gambar4_bobot()]
    for f in figs: plt.close(f)


if __name__ == "__main__":
    for k, v in semua_tabel().items(): print("\n==", k); print(v.to_string(index=False))
    print("\n== angka kunci"); print(angka_kunci().to_string())
    print("\n== RMSE/MAE dan uji DM galat absolut"); print(angka_galat().to_string())
    semua_gambar(); print("\nGambar tersimpan di", OUTN)
