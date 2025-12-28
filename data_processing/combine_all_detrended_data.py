import pandas as pd
from pathlib import Path

orig = pd.read_csv(
    "data/TimeSeries/hourly_delays.csv",
    index_col=0,
    parse_dates=True
)

time_index = orig.index

BASE_DIR = Path("data/Detrended/airports")
series = {}

for airport_dir in BASE_DIR.iterdir():
    z_path = airport_dir / "z_score" / "detrended_series.csv"
    if z_path.exists():
        s = pd.read_csv(z_path)["detrended"].values
        series[airport_dir.name] = pd.Series(s, index=time_index)

df = pd.DataFrame(series, index=time_index)
df.index = pd.to_datetime(df.index)

out = Path("data/Detrended/detrended_z_score.parquet")
df.to_parquet(out)

print("Saved with datetime index:", out)
print(type(df.index))
print(df.index[:5])
