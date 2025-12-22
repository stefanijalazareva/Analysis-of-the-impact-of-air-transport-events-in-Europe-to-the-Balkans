import pandas as pd
from pathlib import Path

BASE_DIR = Path("../results/burr_analysis")

input_file = BASE_DIR / "burr_analysis_summary.csv"
output_pos = BASE_DIR / "burr_positive_delays.csv"
output_neg = BASE_DIR / "burr_negative_delays.csv"

if not input_file.exists():
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pd.read_csv(input_file)

df_pos = df[df["Delay_Type"] == "positive"]
df_neg = df[df["Delay_Type"] == "negative"]

cols = [
    "Airport",
    "Delay_Type",
    "Shape_c",
    "Shape_d",
    "Location",
    "Scale",
    "KS_Statistic",
    "P_value"
]

df_pos = df_pos[cols]
df_neg = df_neg[cols]


# Save separate tables

df_pos.to_csv(output_pos, index=False)
df_neg.to_csv(output_neg, index=False)

print(" Burr tables split successfully")
print(f"   ➜ Positive delays: {output_pos}")
print(f"   ➜ Negative delays: {output_neg}")

