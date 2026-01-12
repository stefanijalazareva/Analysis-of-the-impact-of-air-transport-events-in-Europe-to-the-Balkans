import pandas as pd
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# OpenFlights airports.dat
cols = [
    "id","name","city","country","iata","icao",
    "lat","lon","alt","tz","dst","tz_db","type","source"
]

df = pd.read_csv(
    DATA_DIR / "airports.dat",
    header=None,
    names=cols
)

#only keep the airport ICAO codes that we need
my_airports = [
    'EGLL','LFPG','EHAM','EDDF','LEMD',
    'LEBL','EDDM','EGKK','LIRF','EIDW',
    'LATI','LQSA','LBSF','LBBG','LDZA',
    'LDSP','LDDU','BKPR','LYTV','LWSK'
]

coords = (
    df[df["icao"].isin(my_airports)]
    [["icao","lat","lon"]]
    .rename(columns={"icao":"airport"})
)

coords.to_csv(DATA_DIR / "airport_coordinates.csv", index=False)

print(coords)
print("\nSaved to:", DATA_DIR / "airport_coordinates.csv")
