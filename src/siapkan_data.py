# Tahap 1: curah hujan harian SILO 15 stasiun -> total bulanan + masker pengamatan.
# Sebuah bulan dianggap "teramati" (ok) jika sedikitnya 90% harinya berkode pengamatan langsung SILO (0, 13, 15, 23).
# Bulan hasil interpolasi tetap boleh dipakai sebagai masukan model, tetapi tidak sebagai target latih atau evaluasi.
#   python src/siapkan_data.py
import os, sys, json, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_MENTAH, DATA_OLAHAN

ZIP_SILO = os.path.join(DATA_MENTAH, "SILO_15_stations_rainfall_1926_2025.zip")
KODE_PENGAMATAN = [0, 13, 15, 23]


def siapkan_curah_hujan(zip_path=ZIP_SILO, out_dir=DATA_OLAHAN, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        meta = pd.read_csv(z.open("station_metadata_and_completeness.csv"), dtype={"station_id": str})
        d = pd.read_csv(z.open("rainfall_all_15_stations_1926_2025.csv"), dtype={"station_id": str}, parse_dates=["date"])
    d["obs"] = d["source_code"].isin(KODE_PENGAMATAN).astype(int)
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
        rows.append(dict(stasiun=loc, id=r.station_id, iklim=r.climate_description, lat=r.latitude, lon=r.longitude,
                         cov_1926_1960=cov(1926, 1960), cov_1961_2009=cov(1961, 2009), cov_2010_2014=cov(2010, 2014),
                         cov_2015_2020=cov(2015, 2020), cov_2021_2025=cov(2021, 2025), mean_mm=round(float(np.nanmean(rain)), 1)))
        out["stations"][loc] = {"rain": rain.tolist(), "ok": ok.astype(int).tolist(), "lat": float(r.latitude), "lon": float(r.longitude),
                                "id": r.station_id, "iklim": r.climate_description}
    cov = pd.DataFrame(rows)
    with open(os.path.join(out_dir, "monthly_15.json"), "w", newline="\n") as f: json.dump(out, f)
    cov.to_csv(os.path.join(out_dir, "cakupan_observasi.csv"), index=False, lineterminator="\n")
    if verbose:
        print(f"{len(meta)} stasiun, {len(months)} bulan, {len(d):,} baris harian")
    return cov


if __name__ == "__main__":
    print(siapkan_curah_hujan().to_string(index=False))
