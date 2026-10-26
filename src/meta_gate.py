# Stage 4: local (per-station) meta-learner gate vs cross-station global gate, plus a cross-station transfer test.
# Usage:
#   python src/meta_gate.py build                 -> assemble the meta-dataset of all stations (results/meta_dataset.npz)
#   python src/meta_gate.py local <station>       -> local gates L-G3, L-G4
#   python src/meta_gate.py global <GG3|GG4>      -> global gate with station embedding
#   python src/meta_gate.py transfer <fold 0-4>   -> global gate without the test stations (leave-group-out)
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import nnls

SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
CLIMS = ["Clim-expanding", "Clim-30yr"]
FOLDS = [["Darwin", "Brisbane", "Woomera"], ["Perth", "Cairns", "Birdsville"], ["Tully", "Alice_Springs", "Coffs_Harbour"],
         ["Hobart", "Katherine", "Adelaide"], ["Broome", "Innisfail", "Melbourne"]]


def regime_matrix():
    n34 = IDX["nino34"].values; dmi = IDX["dmi"].values; soi = IDX["soi"].values
    EXO = np.column_stack([np.r_[np.nan, n34[:-1]], np.r_[np.nan, dmi[:-1]], soi])   # row t = values available at the end of month t
    m, s = np.nanmean(EXO[1:VA0], 0), np.nanstd(EXO[1:VA0], 0)
    return (EXO - m) / s


def build():
    Ez = regime_matrix()
    reg = lambda o: np.concatenate([Ez[o - 1], Ez[o - 3:o, 0].mean(keepdims=True), (Ez[o - 1, 0] - Ez[o - 4, 0])[None], Ez[o - 3:o, 1].mean(keepdims=True)])
    rows_all = []; pool = None; static = {}
    sts = [s for s in STATIONS if os.path.exists(os.path.join(OUT, f"base_models_{s}.npz"))]
    for si, st in enumerate(sts):
        B = np.load(os.path.join(OUT, f"base_models_{st}.npz"))
        c30 = np.where(np.isnan(B["clim30"]), B["clim"], B["clim30"])   # fallback: expanding climatology when the 30-year window has no observations
        F = np.concatenate([B["F"], B["clim"][None], c30[None]]); Y, OKm = B["Y"], B["OK"]
        pool = list(B["names"]) + CLIMS; M = len(pool)
        y, ok = load(st); SC = float(y[:VA0][ok[:VA0]].std())
        mon = (np.arange(N) % 12) + 1; ann = np.array([y[:VA0][ok[:VA0] & (mon[:VA0] == m)].mean() for m in range(1, 13)])
        c1 = np.abs(np.sum(ann * np.exp(-2j * np.pi * np.arange(12) / 12))) / ann.sum()       # relative amplitude of the annual harmonic
        static[st] = np.array([np.sin(np.deg2rad(MON["stations"][st]["lat"])), np.cos(np.deg2rad(MON["stations"][st]["lon"])),
                               np.sin(np.deg2rad(MON["stations"][st]["lon"])), np.log(ann.sum()), c1, np.argmax(ann) / 11.0])
        for j, h in enumerate(HORIZONS):
            for T in range(R0 + 11 + 2 * h, N):
                o = T - h + 1; r = o - R0
                E = np.array([[Y[g - h + 1 - R0, j] - F[i, g - h + 1 - R0, j] for i in range(M)] for g in range(o - 12, o)])
                rows_all.append(dict(st=si, j=j, T=T, P=F[:, r, j] / SC, E=E / SC, y=Y[r, j] / SC, ok=bool(OKm[r, j]), SC=SC,
                                     Mo=msc([T])[0], Rg=reg(o), H=np.eye(4)[j]))
        print("build", st, len(rows_all), flush=True)
    D = {k: np.array([r[k] for r in rows_all]) for k in rows_all[0]}
    S = np.array([static[s] for s in sts]); S = (S - S.mean(0)) / (S.std(0) + 1e-9)
    np.savez(os.path.join(OUT, "meta_dataset.npz"), stations=np.array(sts), pool=np.array(pool), static=S, **D)


def load_ds():
    Z = np.load(os.path.join(OUT, "meta_dataset.npz"), allow_pickle=False)
    D = {k: Z[k] for k in Z.files}
    D["split"] = np.where(D["T"] < VA0, 0, np.where(D["T"] < TE0, 1, 2))
    return D


def nnls_w(P, y):
    w, _ = nnls(P, y); return w


def prior(w):
    w = np.maximum(w / max(w.sum(), 1e-9), 0.01); return np.log(w / w.sum())


import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
try: tf.config.experimental.enable_op_determinism()
except AttributeError: pass


def seed(s): np.random.seed(s); tf.keras.utils.set_random_seed(s)


def build_gate(M, n_st, anchor, use_emb, use_static, n_static):
    ei = layers.Input((12, M), name="E"); fi = layers.Input((M,), name="P"); mo = layers.Input((2,), name="Mo")
    rg = layers.Input((6,), name="Rg"); hh = layers.Input((4,), name="H"); ins = [ei, fi, mo, rg, hh]
    parts = [layers.LSTM(16, kernel_regularizer=regularizers.l2(1e-3))(ei), fi, mo, rg, hh]
    if use_emb:
        si = layers.Input((1,), name="st", dtype="int32"); ins.append(si)
        parts.append(layers.Flatten()(layers.Embedding(n_st, 4, embeddings_regularizer=regularizers.l2(1e-3))(si)))
    if use_static:
        sf = layers.Input((n_static,), name="sf"); ins.append(sf); parts.append(sf)
    z = layers.Dropout(0.2)(layers.Concatenate()(parts)); z = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(z)
    if anchor:
        lp = layers.Input((M,), name="LP"); ins.append(lp)
        lg = layers.Add()([layers.Dense(M, kernel_regularizer=regularizers.l2(1e-2), kernel_initializer="zeros")(z), lp])
    else:
        lg = layers.Dense(M)(z)
    w = layers.Softmax(name="w")(lg); m = Model(ins, layers.Dot(axes=1)([w, fi]))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse"); return m


def inputs(D, k, anchor, use_emb, use_static, LP=None):
    X = [D["E"][k], D["P"][k], D["Mo"][k], D["Rg"][k], D["H"][k]]
    if use_emb: X.append(D["st"][k][:, None])
    if use_static: X.append(D["static"][D["st"][k]])
    if anchor: X.append(LP[k])
    return X


def fit_predict(D, tr, va, te, anchor, use_emb, use_static, LP, batch, tag):
    M = D["P"].shape[1]; preds, wts = [], []
    for s_ in SEEDS:
        seed(s_); m = build_gate(M, len(D["stations"]), anchor, use_emb, use_static, D["static"].shape[1])
        m.fit(inputs(D, tr, anchor, use_emb, use_static, LP), D["y"][tr], epochs=300, batch_size=batch, verbose=0,
              validation_data=(inputs(D, va, anchor, use_emb, use_static, LP), D["y"][va]),
              callbacks=[callbacks.EarlyStopping(patience=30, restore_best_weights=True)])
        wm = Model(m.inputs, m.get_layer("w").output)
        preds.append(m.predict(inputs(D, te, anchor, use_emb, use_static, LP), verbose=0, batch_size=2048).ravel())
        wts.append(wm.predict(inputs(D, te, anchor, use_emb, use_static, LP), verbose=0, batch_size=2048))
        print(tag, "seed", s_, "done", flush=True)
    return np.array(preds), np.array(wts)


def station_nnls_prior(D):
    """NNLS weights per station-horizon from the observed months before the test period (1962-2014)."""
    LP = np.zeros_like(D["P"]); W = {}
    for si in range(len(D["stations"])):
        for j in range(4):
            k = (D["st"] == si) & (D["j"] == j); kt = k & (D["split"] < 2) & D["ok"]
            w = nnls_w(D["P"][kt], D["y"][kt]); W[(si, j)] = w; LP[k] = prior(w)
    return LP, W


if __name__ == "__main__":
    cmd = sys.argv[1]; t0 = time.time()
    if cmd == "build":
        build(); sys.exit()
    D = load_ds(); LP, W = station_nnls_prior(D)
    base_tr = (D["split"] == 0) & D["ok"]; base_va = (D["split"] == 1) & D["ok"]
    if cmd == "local":
        st = sys.argv[2]; si = list(D["stations"]).index(st); ks = D["st"] == si; te = np.where(ks & (D["split"] == 2))[0]
        out = {"idx": te}
        tr_i, va_i = np.where(ks & base_tr)[0], np.where(ks & base_va)[0]
        if len(va_i) == 0:
            # station without observed months in 2010-2014: early stopping uses the last 20% of observed training targets
            Tcut = np.quantile(D["T"][tr_i], 0.8); va_i = tr_i[D["T"][tr_i] > Tcut]; tr_i = tr_i[D["T"][tr_i] <= Tcut]
            print(st, "empty validation set; early stopping uses the last", len(va_i), "training samples (target >", int(Tcut), ")", flush=True)
        for name, anc in [("L-G3", False), ("L-G4", True)]:
            p, w = fit_predict(D, tr_i, va_i, te, anc, False, False, LP, 64, f"{st} {name}")
            out[name] = p; out[name + "_w"] = w.mean(0)
        np.savez(os.path.join(OUT, f"pred_local_{st}.npz"), **out)
    elif cmd == "global":
        name = sys.argv[2]; anc = name == "GG4"; te = np.where(D["split"] == 2)[0]
        p, w = fit_predict(D, np.where(base_tr)[0], np.where(base_va)[0], te, anc, True, True, LP, 256, name)
        np.savez(os.path.join(OUT, f"pred_global_{name}.npz"), idx=te, pred=p, w=w.mean(0))
    elif cmd == "transfer":
        f = int(sys.argv[2]); hold = [list(D["stations"]).index(s) for s in FOLDS[f] if s in D["stations"]]
        inh = np.isin(D["st"], hold)
        # transfer anchor: pooled NNLS from the training stations (the test stations are not used at all)
        LPt = np.zeros_like(D["P"])
        for j in range(4):
            kt = (~inh) & (D["j"] == j) & (D["split"] < 2) & D["ok"]; LPt[D["j"] == j] = prior(nnls_w(D["P"][kt], D["y"][kt]))
        te = np.where(inh & (D["split"] == 2))[0]
        p, w = fit_predict(D, np.where((~inh) & base_tr)[0], np.where((~inh) & base_va)[0], te, True, False, True, LPt, 256, f"transfer fold {f}")
        np.savez(os.path.join(OUT, f"pred_transfer_{f}.npz"), idx=te, pred=p, w=w.mean(0))
    print(cmd, sys.argv[2:], "done", round((time.time() - t0) / 60, 1), "min", flush=True)
