# Data

## `raw/` — raw data

| File | Contents | Source |
|---|---|---|
| `SILO_15_stations_rainfall_1926_2025.zip` | Daily rainfall for 15 stations, 1 Jan 1926 – 31 Dec 2025, with the source code of every day, station metadata, and the original JSON responses | Queensland Government, SILO Patched Point Dataset (downloaded 25 October 2026) |
| `indices/ersst_v5_ncei_1921_1939.zip`, `indices/ersst_v5_ncei_1940_1960.zip` | 480 monthly ERSST v5 grid files (netCDF), Jan 1921 – Dec 1960, split into two archives so that each file is < 25 MB | NOAA NCEI, Extended Reconstructed SST v5 |
| `indices/ersst5_nino_cpc.ascii` | Niño indices (including Niño3.4) from ERSST v5, 1950 – present | NOAA Climate Prediction Center |
| `indices/dmi_had.txt` | Dipole Mode Index from HadISST | NOAA Physical Sciences Laboratory |
| `indices/soi_cru.dat` | Southern Oscillation Index | Climatic Research Unit, University of East Anglia |

SILO source codes: 0 = official BoM observation; 23 = BoM observation from a neighbouring station; 13 and 15 = accumulated rainfall redistributed; 25, 35, 75 = interpolated. This study treats codes 0, 13, 15, and 23 as observations. Full details are in the `README.md` inside the SILO archive.

## `processed/` — derived data (rebuilt by the notebook, Sections 1–2)

| File | Contents | Produced by |
|---|---|---|
| `monthly_15.json` | Monthly rainfall totals for 15 stations (1,200 months) and the `ok` mask (months with ≥ 90% of days carrying an observation code) | `src/prepare_data.py` |
| `observation_coverage.csv` | Proportion of observed months per station and period, coordinates, and mean rainfall | `src/prepare_data.py` |
| `climate_indices_1926_2025.csv` | Monthly Niño3.4 (anomaly relative to 1926–1960), DMI, and SOI | `src/climate_indices.py` |

## SHA-256 checksums

```
9d0d0139d7c6eb529153aa6c2290b0201bd81674005f39f0588c80035a018caa  raw/SILO_15_stations_rainfall_1926_2025.zip
e0c22137f86c4c0e9cc36ed54431d6f6596cba88107a73ba4033806b64bc94d7  raw/indices/ersst_v5_ncei_1921_1939.zip
d9f37c7c438a34ebf8b52b6ddf7fbd958e1dcb102324b24f1a9d44d8baaa4e8e  raw/indices/ersst_v5_ncei_1940_1960.zip
bedcd0683c07f7ccb2ae358a90cb39d57413dcc8881027e0c30a1811f6d58012  raw/indices/ersst5_nino_cpc.ascii
42c1000e221addacac4451b8b9ddfbc5ee732b99ee69b0be5105e69ebe881e05  raw/indices/dmi_had.txt
483808d2b31adbc2f114819e51876041bb0c5ffe323e6eb48149b2f69ba179cb  raw/indices/soi_cru.dat
d812682de7c5416929189276588fc269647c414dfe88cf1526db7113deb224a7  processed/monthly_15.json
61c0df1f06597bc98d2120e34da21cf29d52fd5cb8073f3ae5c8ef3b907f93bd  processed/observation_coverage.csv
af0d72a0ec81d5d0c37f556a21dbca20c8bee8c3100ada19f9aea3b5c621d220  processed/climate_indices_1926_2025.csv
```

## Licence and attribution

- **SILO**: © Queensland Government, Creative Commons Attribution 4.0 (CC BY 4.0). Built from Australian Bureau of Meteorology observations. https://www.longpaddock.qld.gov.au/silo/ — Jeffrey et al. (2001), *Environmental Modelling & Software* 16(4), 309–330, doi:10.1016/S1364-8152(01)00008-1.
- **ERSST v5**: NOAA NCEI. Huang et al. (2017), *Journal of Climate* 30(20), 8179–8205, doi:10.1175/JCLI-D-16-0836.1.
- **HadISST / DMI**: Met Office Hadley Centre; DMI series from NOAA PSL. Rayner et al. (2003), *JGR Atmospheres* 108(D14), 4407, doi:10.1029/2002JD002670; Saji et al. (1999), *Nature* 401, 360–363, doi:10.1038/43854.
- **SOI**: Climatic Research Unit, University of East Anglia.

SILO states that its products are not intended for climate-change detection studies. A complete series is not necessarily homogeneous, nor are all of its values direct observations.
