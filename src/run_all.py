# Runs the training pipeline in parallel (one process per station or per gate).
#   python src/run_all.py base   -> rolling base models, 15 stations
#   python src/run_all.py meta   -> meta-dataset, global gates, transfer gates, local gates
#   python src/run_all.py eval   -> evaluation and statistical tests (NSE, RMSE, MAE, DM tests)
#   python src/run_all.py all    -> all three in sequence
# The number of parallel processes is set with the environment variable WORKERS (default: CPU cores - 1).
# Existing output files are skipped; delete them from results/ to retrain.
import os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor
SRC = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(SRC); PY = sys.executable
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results")); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(ROOT, "log"); os.makedirs(LOG, exist_ok=True)
WORKERS = int(os.environ.get("WORKERS", max(1, (os.cpu_count() or 2) - 1)))
STATIONS = ["Darwin", "Katherine", "Broome", "Cairns", "Innisfail", "Tully", "Brisbane", "Coffs_Harbour", "Alice_Springs",
            "Woomera", "Birdsville", "Perth", "Adelaide", "Melbourne", "Hobart"]


def job(args, logname, done=None):
    if done and os.path.exists(os.path.join(OUT, done)):
        return logname, "already exists, skipped"
    t = time.time()
    with open(os.path.join(LOG, logname + ".txt"), "w", encoding="utf-8") as f:
        r = subprocess.run([PY, os.path.join(SRC, args[0])] + args[1:], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    return logname, f"exit code {r.returncode}, {round((time.time() - t) / 60, 1)} min"


def parallel(jobs):
    with ThreadPoolExecutor(WORKERS) as ex:
        for name, status in ex.map(lambda j: job(*j), jobs):
            print(time.strftime("%H:%M:%S"), name, status, flush=True)


def run(phase="all"):
    if phase in ("base", "all"):
        print("Base models:", len(STATIONS), "stations,", WORKERS, "parallel processes", flush=True)
        parallel([(["base_models_rolling.py", s], f"base_{s}", f"base_models_{s}.npz") for s in STATIONS])
    if phase in ("meta", "all"):
        print(*job(["meta_gate.py", "build"], "meta_build"), flush=True)
        jobs = [(["meta_gate.py", "global", g], f"global_{g}", f"pred_global_{g}.npz") for g in ["GG4", "GG3"]]
        jobs += [(["meta_gate.py", "transfer", str(f)], f"transfer_{f}", f"pred_transfer_{f}.npz") for f in range(5)]
        jobs += [(["meta_gate.py", "local", s], f"local_{s}", f"pred_local_{s}.npz") for s in STATIONS]
        parallel(jobs)
    if phase in ("eval", "all"):
        print(*job(["evaluate.py"], "eval"), flush=True)
        print(*job(["evaluate_errors.py"], "eval_errors"), flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
