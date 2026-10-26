# Stage 1: SILO daily rainfall for 15 stations -> monthly totals + observation mask.
# A month is treated as "observed" (ok) if at least 90% of its days carry a direct SILO observation code (0, 13, 15, 23).
# Interpolated months may still be used as model inputs, but not as training or evaluation targets.
#   python src/prepare_data.py
import os, sys, json, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_RAW, DATA_PROCESSED

ZIP_SILO = os.path.join(DATA_RAW, "SILO_15_stations_rainfall_1926_2025.zip")
OBSERVATION_CODES = [0, 13, 15, 23]


def prepare_rainfall(zip_path=ZIP_SILO, out_dir=DATA_PROCESSED, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        meta = pd.read_csv(z.open("station_metadata_and_completeness.csv"), dtype={"station_id": str})
        d = pd.read_csv(z.open("rainfall_all_15_stations_1926_2025.csv"), dtype={"station_id": str}, parse_dates=["date"])
    d["obs"] = d["source_code"].isin(OBSERVATION_CODES).astype(int)
    d["ym"] = d["date"].dt.to_period("M")
    g = d.groupby(["location", "ym"]).agg(rain=("rainfall_mm", "sum"), obsfrac=("obs", "mean"), n=("obs", "size")).reset_index()
    months = pd.period_range("1926-01", "2025-12", freq="M")
    out = {"months": [str(m) for m in months], "stations": {}}
    rows = []
    yrs = np.array([m.year for m in months])
    for loc in meta["location"]:
        s = g[g.location == loc].set_index("ym").reindex(months)
        rain = s["rain"].values.astype(float); of = s["obsfrac"].values.astype(float)
        ok = of >= 0.9
        def cov(a, b): k = (yrs >= a) & (yrs <= b); return round(float(ok[k].mean()), 3)
        r = meta[meta.location == loc].iloc[0]
        rows.append(dict(station=loc, id=r.station_id, climate=r.climate_description, lat=r.latitude, lon=r.longitude,
                         cov_1926_1960=cov(1926, 1960), cov_1961_2009=cov(1961, 2009), cov_2010_2014=cov(2010, 2014),
                         cov_2015_2020=cov(2015, 2020), cov_2021_2025=cov(2021, 2025), mean_mm=round(float(np.nanmean(rain)), 1)))
        out["stations"][loc] = {"rain": rain.tolist(), "ok": ok.astype(int).tolist(), "lat": float(r.latitude), "lon": float(r.longitude),
                                "id": r.station_id, "climate": r.climate_description}
    cov = pd.DataFrame(rows)
    with open(os.path.join(out_dir, "monthly_15.json"), "w", newline="\n") as f: json.dump(out, f)
    cov.to_csv(os.path.join(out_dir, "observation_coverage.csv"), index=False, lineterminator="\n")
    if verbose:
        print(f"{len(meta)} stations, {len(months)} months, {len(d):,} daily rows")
    return cov


if __name__ == "__main__":
    print(prepare_rainfall().to_string(index=False))
