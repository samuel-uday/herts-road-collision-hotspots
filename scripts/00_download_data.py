"""Download the raw DfT STATS19 files and the code guide into data/raw/.

Run from the project root:  .venv/bin/python scripts/00_download_data.py
Source: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data (Open Government Licence)
"""
from pathlib import Path
from urllib.request import urlretrieve

RAW = Path("data/raw")
BASE = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-{}-last-5-years.csv"
GUIDE = ("https://assets.publishing.service.gov.uk/media/6ab2a71d997a4b2950cced58/"
         "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx")

RAW.mkdir(parents=True, exist_ok=True)
files = {f"{t}-last-5-years.csv": BASE.format(t) for t in ("collision", "vehicle", "casualty")}
files["data-guide-2025.xlsx"] = GUIDE

for name, url in files.items():
    target = RAW / name
    if target.exists():
        print(f"✓ {name} already present")
        continue
    print(f"↓ {name} …", flush=True)
    urlretrieve(url, target)
    print(f"  {target.stat().st_size / 1e6:.0f} MB")
