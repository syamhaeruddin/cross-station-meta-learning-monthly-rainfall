# Stage 3: base models with rolling retraining every 5 years from 1961, for one SILO station.
#   python src/base_models_rolling.py <station>      (example: python src/base_models_rolling.py Darwin)
# Training and evaluation targets are observed months only (>= 90% of days with an observation code); interpolated months are still used as inputs.
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from vmdpy import VMD
from sklearn.svm import SVR
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
try: tf.config.experimental.enable_op_determinism()
except AttributeError: pass
def seed(s): np.random.seed(s); tf.keras.utils.set_random_seed(s)
def dom_freq(s):
    s = s - s.mean(); p = np.abs(np.fft.rfft(s))**2; f = np.fft.rfftfreq(len(s)); p[0] = 0; return f[np.argmax(p)]
def grouping(imfs):
    order = np.argsort([dom_freq(i) for i in imfs], kind="stable"); lo, mi, hi = np.array_split(order, 3)
    mp = {int(k): "L" for k in lo}; mp.update({int(k): "M" for k in mi}); mp.update({int(k): "H" for k in hi}); return mp
def bands3(imfs, mp):
    out = np.zeros((3, imfs.shape[1]))
    for k in range(imfs.shape[0]): out[{"L": 0, "M": 1, "H": 2}[mp.get(k, "L")]] += imfs[k]
    return out
def atrous(x, J=5):
    c = np.asarray(x, float).copy(); comps = []
    for j in range(1, J + 1):
        s_ = 2**(j - 1); prev = np.concatenate([np.full(s_, c[0]), c[:-s_]]); cn = 0.5 * (c + prev); comps.append(c - cn); c = cn
    comps.append(c); return np.array(comps)
def vmd(x):
    x = np.asarray(x, float)[len(x) % 2:]; u, _, _ = VMD(x, 2000, 0, 6, 0, 1, 1e-7); return u
def masked_mse(yt, yp):
    m = tf.math.is_finite(yt); yt0 = tf.where(m, yt, tf.zeros_like(yt)); mf = tf.cast(m, yp.dtype)
    return tf.reduce_sum(mf * tf.square(yt0 - yp)) / tf.maximum(tf.reduce_sum(mf), 1.0)

st = sys.argv[1]; y, ok = load(st); t0 = time.time()
n34 = IDX["nino34"].values; dmi = IDX["dmi"].values; soi = IDX["soi"].values
# release latency: Nino3.4 and DMI are available up to month o-2, SOI up to o-1 (o = first target month)
EXO = np.column_stack([np.r_[np.nan, n34[:-1]], np.r_[np.nan, dmi[:-1]], soi])   # row t holds the values available at the end of month t
REFITS = list(range(R0, N, STEP)); ORIG = np.arange(R0, N)
# causal a trous Haar wavelet; band grouping fixed from data before 1961
full_w = atrous(y); mpW = grouping(atrous(y[:R0])); bW_full = bands3(full_w, mpW)
LAST = {"VMD": np.zeros((N, 3, 12)), "WT": np.zeros((N, 3, 12))}
for t in range(T0, N):
    u = vmd(y[:t]); LAST["VMD"][t] = bands3(u, grouping(u))[:, -12:]; LAST["WT"][t] = bW_full[:, t - 12:t]
print(st, "decomposition", round(time.time() - t0), "s", flush=True)
H = HORIZONS; HM = max(H)
tgt = lambda t, h: (y[t + h - 1] if (t + h - 1 < N and ok[t + h - 1]) else np.nan)
NAMES = ["SARIMA", "SVR", "XGBoost", "XGBoost-VMD", "XGBoost-WT", "LSTM", "GRU", "LSTM-VMD", "LSTM-WT"]
FC = {k: np.full((len(ORIG), len(H)), np.nan) for k in NAMES}
for s in REFITS:
    O = np.arange(s, min(s + STEP, N)); io = O - R0
    em, es = np.nanmean(EXO[T0:s], 0), np.nanstd(EXO[T0:s], 0); Ez = (EXO - em) / es
    yo = y[:s][ok[:s]]; ym, ys = yo.mean(), yo.std()
    reg = lambda t: Ez[t - 6:t].ravel()                       # 3 indices x 6 lags
    feat_raw = lambda t, h: np.concatenate([(y[t - 12:t] - ym) / ys, reg(t), msc([t + h - 1])[0]])
    feat_hyb = lambda m, t, h: np.concatenate([LAST[m][t].ravel() / ys, reg(t), msc([t + h - 1])[0]])
    seq_raw = lambda t: np.column_stack([(y[t - 12:t] - ym) / ys, Ez[t - 12:t], msc(np.arange(t - 12, t))])
    seq_hyb = lambda m, t: np.column_stack([LAST[m][t].T / ys, Ez[t - 12:t], msc(np.arange(t - 12, t))])
    sm = SARIMAX(y[:s], order=(2, 1, 2), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    FC["SARIMA"][io] = np.array([[sm.apply(y[:t]).forecast(HM)[h - 1] for h in H] for t in O])
    def tab(kind, featf):
        out = np.zeros((len(O), len(H)))
        for j, h in enumerate(H):
            tr = np.array([t for t in range(T0, s - h + 1) if ok[t + h - 1]])
            Xt = np.array([featf(t, h) for t in tr]); yt = (np.array([y[t + h - 1] for t in tr]) - ym) / ys
            m = SVR(C=1.0, epsilon=0.1).fit(Xt, yt) if kind == "SVR" else XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=1).fit(Xt, yt)
            out[:, j] = m.predict(np.array([featf(t, h) for t in O])) * ys + ym
        return out
    FC["SVR"][io] = tab("SVR", feat_raw); FC["XGBoost"][io] = tab("XGB", feat_raw)
    for meth in ["VMD", "WT"]: FC[f"XGBoost-{meth}"][io] = tab("XGB", lambda t, h, m=meth: feat_hyb(m, t, h))
    def rnn(seqf, cell):
        tr = np.arange(T0, s - HM + 1); Ys = (np.array([[tgt(t, h) for h in H] for t in tr]) - ym) / ys
        keep = np.isfinite(Ys).any(1); tr, Ys = tr[keep], Ys[keep]; Xs = np.array([seqf(t) for t in tr])
        seed(42); inp = layers.Input(Xs.shape[1:]); z = (layers.LSTM if cell == "LSTM" else layers.GRU)(64)(inp)
        z = layers.Dropout(0.2)(z); z = layers.Dense(32, activation="relu")(z); m = Model(inp, layers.Dense(len(H))(z)); m.compile(optimizer="adam", loss=masked_mse)
        m.fit(Xs, Ys, epochs=200, batch_size=32, validation_split=0.15, verbose=0, callbacks=[callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
        return m.predict(np.array([seqf(t) for t in O]), verbose=0) * ys + ym
    FC["LSTM"][io] = rnn(seq_raw, "LSTM"); FC["GRU"][io] = rnn(seq_raw, "GRU")
    for meth in ["VMD", "WT"]: FC[f"LSTM-{meth}"][io] = rnn(lambda t, m=meth: seq_hyb(m, t), "LSTM")
    print(st, "refit", year(s), round((time.time() - t0) / 60, 1), "min", flush=True)
Yfull = np.array([[y[t + h - 1] if t + h - 1 < N else np.nan for h in H] for t in ORIG])
OKm = np.array([[bool(ok[t + h - 1]) if t + h - 1 < N else False for h in H] for t in ORIG])
mon = (np.arange(N) % 12) + 1
# rolling climatologies: (a) expanding, mean of the same calendar month over all observed months before the origin; (b) 30-year window
clim = np.array([[y[:t][(mon[:t] == ((t + h - 1) % 12) + 1) & ok[:t]].mean() for h in H] for t in ORIG])
clim30 = np.array([[y[max(0, t - 360):t][(mon[max(0, t - 360):t] == ((t + h - 1) % 12) + 1) & ok[max(0, t - 360):t]].mean() for h in H] for t in ORIG])
np.savez(os.path.join(OUT, f"base_models_{st}.npz"), names=np.array(NAMES), F=np.stack([FC[k] for k in NAMES]), Y=Yfull, OK=OKm,
         clim=clim, clim30=clim30, orig=ORIG, H=np.array(H))
print(st, "done", round((time.time() - t0) / 60, 1), "min", flush=True)
