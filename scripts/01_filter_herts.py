"""Filter the national DfT STATS19 files down to Hertfordshire.

Hertfordshire Constabulary is police force 41. Using the police force (not the
highway authority) keeps motorway collisions on the M1, M25 and A1(M), which
National Highways manages rather than the county council.

Run from the project root:  .venv/bin/python scripts/01_filter_herts.py
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
CLEAN = Path("data/clean")
HERTS_POLICE_FORCE = 41

CLEAN.mkdir(parents=True, exist_ok=True)

collisions = pd.read_csv(RAW / "collision-last-5-years.csv", low_memory=False)
herts = collisions[collisions["police_force"] == HERTS_POLICE_FORCE]
ids = set(herts["collision_index"])

vehicles = pd.read_csv(RAW / "vehicle-last-5-years.csv", low_memory=False)
casualties = pd.read_csv(RAW / "casualty-last-5-years.csv", low_memory=False)

herts_vehicles = vehicles[vehicles["collision_index"].isin(ids)]
herts_casualties = casualties[casualties["collision_index"].isin(ids)]

herts.to_csv(CLEAN / "herts_collisions_raw_codes.csv", index=False)
herts_vehicles.to_csv(CLEAN / "herts_vehicles_raw_codes.csv", index=False)
herts_casualties.to_csv(CLEAN / "herts_casualties_raw_codes.csv", index=False)

print(f"Collisions: {len(herts):,}")
print(f"Vehicles:   {len(herts_vehicles):,}")
print(f"Casualties: {len(herts_casualties):,}")
