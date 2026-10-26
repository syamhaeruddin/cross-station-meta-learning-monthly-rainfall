# Stage 2: monthly climate indices 1926-2025 from sources that are consistent over the whole period.
#   Nino3.4 : ERSST v5. Box mean 5N-5S, 190-240E (cosine-latitude weighted) computed from the NCEI grid files
#             for 1926-1949, then the CPC series (also ERSST v5) from 1950. Anomalies relative to the 1926-1960 climatology.
#   DMI     : Dipole Mode Index from HadISST (NOAA PSL).
#   SOI     : Southern Oscillation Index, Climatic Research Unit (CRU).
#   python src/climate_indices.py
import os, sys, re, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_RAW, DATA_PROCESSED

D_IDX = os.path.join(DATA_RAW, "indices")


def nino34_ersst_grid(zip_paths=None):
    """Absolute Nino3.4 from the monthly ERSST v5 grid files (one netCDF file per month) inside the zip archives
    (ersst_v5_ncei_1921_1939.zip and ersst_v5_ncei_1940_1960.zip; split so that each file is < 25 MB)."""
    import netCDF4, glob
    ab = {}
    for zip_path in (zip_paths or sorted(glob.glob(os.path.join(D_IDX, "ersst_v5_ncei_*.zip")))):
        with zipfile.ZipFile(zip_path) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".nc")):
                ym = re.search(r"(\d{6})\.nc$", name).group(1)
                ds = netCDF4.Dataset("mem", memory=z.read(name))
                lat = ds["lat"][:]; lon = ds["lon"][:]; sst = np.ma.squeeze(ds["sst"][:])
                la = (lat >= -5) & (lat <= 5); lo = (lon >= 190) & (lon <= 240)
                sub = sst[np.ix_(la, lo)]; w = np.cos(np.deg2rad(lat[la]))[:, None] * np.ones(lo.sum())[None]
                ab[pd.Period(f"{ym[:4]}-{ym[4:]}", "M")] = float(np.ma.average(sub, weights=np.ma.array(w, mask=np.ma.getmaskarray(sub))))
                ds.close()
    return pd.Series(ab)


def psl(path):
    """Reads a PSL/CRU-format text file: one row per year, 12 monthly values; values <= -9.9 are treated as missing."""
    v = {}
    for line in open(path):
        t = line.split()
        if len(t) >= 13 and re.fullmatch(r"\d{4}", t[0]):
            for m in range(12):
                x = float(t[m + 1]); v[pd.Period(f"{t[0]}-{m+1:02d}", "M")] = np.nan if x <= -9.9 else x
    return pd.Series(v)


def compute_indices(out_dir=DATA_PROCESSED, verbose=True):
    months = pd.period_range("1926-01", "2025-12", freq="M")
    ab = nino34_ersst_grid()
    cpc = pd.read_csv(os.path.join(D_IDX, "ersst5_nino_cpc.ascii"), sep=r"\s+")
    cpc.index = [pd.Period(f"{int(y)}-{int(m):02d}", "M") for y, m in zip(cpc.YR, cpc.MON)]
    c34 = cpc["NINO3.4"]
    ov = ab.index.intersection(c34.index)
    if verbose:
        print("Nino3.4 grid vs CPC, overlap", len(ov), "months: r =", round(np.corrcoef(ab[ov], c34[ov])[0, 1], 5),
              "| max difference =", round(float(np.max(np.abs(ab[ov] - c34[ov]))), 3), "degC")
    n34abs = pd.concat([ab[ab.index < pd.Period("1950-01", "M")], c34]).reindex(months)
    clim = n34abs[(n34abs.index.year >= 1926) & (n34abs.index.year <= 1960)].groupby(lambda p: p.month).mean()
    n34 = n34abs - np.array([clim[p.month] for p in months])
    dmi = psl(os.path.join(D_IDX, "dmi_had.txt")).reindex(months)
    soi = psl(os.path.join(D_IDX, "soi_cru.dat")).reindex(months)
    df = pd.DataFrame({"nino34": n34.values, "dmi": dmi.values, "soi": soi.values}, index=months.astype(str))
    os.makedirs(out_dir, exist_ok=True); df.to_csv(os.path.join(out_dir, "climate_indices_1926_2025.csv"), lineterminator="\n")
    if verbose:
        print("missing values:", df.isna().sum().to_dict())
    return df


if __name__ == "__main__":
    print(compute_indices().describe().round(2).to_string())
