# Hertfordshire Road Collision Hotspots (2021–2025)

Where and when do road collisions happen in Hertfordshire, and what makes them more likely to kill or seriously injure (KSI) someone?

**Data:** Department for Transport STATS19 road safety open data · ONS district boundaries · OpenStreetMap place names. All under open licences.

## Project structure

```
data/raw/        national DfT files + data guide (downloaded)
data/clean/      Hertfordshire-only, cleaned and labelled data
data/geo/        district boundaries, place-name cache
scripts/         00_download_data.py, 01_filter_herts.py: get the data and cut it down to Hertfordshire
notebooks/       01_cleaning → 02_eda → 03_hotspots
app/             dashboard.py: interactive Streamlit dashboard
report/          figures, findings and the interactive hotspot map
```

## Run it

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/00_download_data.py       # ~250 MB of DfT files into data/raw/
.venv/bin/python scripts/01_filter_herts.py        # cut down to Hertfordshire
.venv/bin/jupyter lab                              # run notebooks 01 → 03 in order
.venv/bin/streamlit run app/dashboard.py           # open http://localhost:8501
```

## Methods

- **Cleaning:** decoded with the DfT data guide; `-1` treated as missing; vehicle and casualty tables rolled up to one row per collision.
- **EDA:** KSI rates with 95% Wilson confidence intervals; chi-square tests with Cramér's V; per-resident district rates (Census 2021).
- **Hotspots:** DBSCAN (100 m, 8+ collisions) for repeat sites; Getis-Ord Gi\* on a 500 m grid with FDR correction; severity score slight 1 / serious 3 / fatal 5.

## Key findings

- 8,375 injury collisions in 2021–25; 20.9% killed or seriously injured someone.
- Pedestrians, cyclists and motorcyclists are 28% of casualties but 52% of KSI casualties.
- Night-time (00–05) collisions are the most severe (33% KSI); 60 mph rural roads are the most dangerous speed limit.
- Gi\* hot spots cover 11% of the county but contain 47% of KSI collisions.
- Top repeat site: A10 at Churchgate, Broxbourne (78 collisions, 10 KSI). All top-20 sites had collisions in every year.

## Limitations

Counts aren't adjusted for traffic volume; motorways are managed by National Highways; clusters show *where*, not *why*. A site check is needed before recommending fixes.
