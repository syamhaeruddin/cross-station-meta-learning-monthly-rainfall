# Constants, file locations, and shared functions for the whole pipeline.
# All locations are relative to the repository root, so the repository can be run from any folder.
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]:
    os.environ[v] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"; os.environ["PYTHONHASHSEED"] = "0"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd
from scipy import stats

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results")); os.makedirs(OUT, exist_ok=True)
REPORTS = os.path.join(ROOT, "reports")

N = 1200                      # Jan 1926 .. Dec 2025
T0 = 180                      # first training target Jan 1941
R0, STEP = 420, 60            # first refit Jan 1961, then every 60 months (13 refits, last Jan 2021)
MT0, VA0, TE0 = 432, 1008, 1068   # meta-training from 1962, meta-validation Jan 2010, test Jan 2015
HORIZONS = [1, 3, 6, 12]
STATIONS_ALL = ["Darwin", "Katherine", "Broome", "Cairns", "Innisfail", "Tully", "Brisbane", "Coffs_Harbour", "Alice_Springs",
                "Woomera", "Birdsville", "Perth", "Adelaide", "Melbourne", "Hobart"]

_MON = None; _IDX = None
def _load_data():
    global _MON, _IDX
    if _MON is None:
        _MON = json.load(open(os.path.join(DATA_PROCESSED, "monthly_15.json")))
        _IDX = pd.read_csv(os.path.join(DATA_PROCESSED, "climate_indices_1926_2025.csv"), index_col=0)
    return _MON, _IDX
class _Lazy:
    """Loads the processed data on first use, so the module can be imported before the data are prepared."""
    def __init__(self, i): self._i = i
    def __getattr__(self, a): return getattr(_load_data()[self._i], a)
    def __getitem__(self, k): return _load_data()[self._i][k]
    def __iter__(self): return iter(_load_data()[self._i])
    def __contains__(self, k): return k in _load_data()[self._i]
MON = _Lazy(0)
IDX = _Lazy(1)
STATIONS = STATIONS_ALL


def load(st):
    s = MON["stations"][st]
    return np.array(s["rain"], float), np.array(s["ok"], bool)
def year(t): return 1926 + t // 12
def msc(idx):
    m = (np.asarray(idx) % 12) + 1
    return np.column_stack([np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])
def metrics(a, p):
    a, p = np.asarray(a), np.asarray(p); e = a - p
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))), "NSE": float(1 - np.sum(e**2) / np.sum((a - a.mean())**2))}
def dm_test(e1, e2, h=1):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction; long-run variance up to lag h-1."""
    d = e1**2 - e2**2; n = len(d); dc = d - d.mean()
    gam = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    v = gam[0] + 2 * sum(gam[1:]); v = v if v > 0 else gam[0]
    stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"DM": float(stat), "p": float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))}
def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj
