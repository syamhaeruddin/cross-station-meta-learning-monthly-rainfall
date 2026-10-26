# Menjalankan pipeline pelatihan secara paralel (satu proses per stasiun atau per gate).
#   python src/run_all.py base   -> model dasar bergulir 15 stasiun
#   python src/run_all.py meta   -> dataset meta, gate global, gate transfer, gate lokal
#   python src/run_all.py eval   -> evaluasi dan uji statistik (NSE, RMSE, MAE, uji DM)
#   python src/run_all.py all    -> ketiganya berurutan
# Jumlah proses paralel diatur dengan variabel lingkungan WORKERS (bawaan: jumlah inti CPU - 1).
# Berkas hasil yang sudah ada dilewati; hapus berkasnya di folder hasil/ untuk melatih ulang.
import os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor
SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC); PY = sys.executable
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "hasil")); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(ROOT, "log"); os.makedirs(LOG, exist_ok=True)
WORKERS = int(os.environ.get("WORKERS", max(1, (os.cpu_count() or 2) - 1)))
STATIONS = ["Darwin", "Katherine", "Broome", "Cairns", "Innisfail", "Tully", "Brisbane", "Coffs_Harbour", "Alice_Springs",
            "Woomera", "Birdsville", "Perth", "Adelaide", "Melbourne", "Hobart"]


def job(args, logname, done=None):
    if done and os.path.exists(os.path.join(OUT, done)):
        return logname, "sudah ada, dilewati"
    t = time.time()
    with open(os.path.join(LOG, logname + ".txt"), "w", encoding="utf-8") as f:
        r = subprocess.run([PY, os.path.join(SRC, args[0])] + args[1:], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    return logname, f"kode {r.returncode}, {round((time.time() - t) / 60, 1)} menit"


def parallel(jobs):
    with ThreadPoolExecutor(WORKERS) as ex:
        for name, status in ex.map(lambda j: job(*j), jobs):
            print(time.strftime("%H:%M:%S"), name, status, flush=True)


def run(phase="all"):
    if phase in ("base", "all"):
        print("Model dasar:", len(STATIONS), "stasiun,", WORKERS, "proses paralel", flush=True)
        parallel([(["base_roll.py", s], f"base_{s}", f"model_dasar_{s}.npz") for s in STATIONS])
    if phase in ("meta", "all"):
        print(*job(["meta_global.py", "build"], "meta_build"), flush=True)
        jobs = [(["meta_global.py", "global", g], f"global_{g}", f"pred_global_{g}.npz") for g in ["GG4", "GG3"]]
        jobs += [(["meta_global.py", "transfer", str(f)], f"transfer_{f}", f"pred_transfer_{f}.npz") for f in range(5)]
        jobs += [(["meta_global.py", "local", s], f"local_{s}", f"pred_local_{s}.npz") for s in STATIONS]
        parallel(jobs)
    if phase in ("eval", "all"):
        print(*job(["eval_global.py"], "eval"), flush=True)
        print(*job(["eval_galat.py"], "eval_galat"), flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
