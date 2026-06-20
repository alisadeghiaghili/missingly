import numpy as np
import pandas as pd

np.random.seed(42)

rows = []

timestamps = pd.date_range(
    "2024-01-01",
    periods=100,
    freq="1min"
)

sensor_ids = [
    "TK001",
    "TK002",
    "TK003",
    "TK004",
    "TK005",
]

for ts in timestamps:
    for sensor in sensor_ids:

        rows.append(
            {
                "timestamp": ts,
                "sensor_id": sensor,
                "temperature": np.random.normal(45, 2),
                "pressure": np.random.normal(3.5, 0.2),
                "flow": np.random.normal(110, 5),
                "ph": np.random.normal(7, 0.3),
                "status": np.random.choice(
                    ["open", "closed"]
                ),
            }
        )

df = pd.DataFrame(rows)

# حدود 10٪ Missing
for col in [
    "temperature",
    "pressure",
    "flow",
    "ph",
]:
    mask = np.random.rand(len(df)) < 0.10
    df.loc[mask, col] = np.nan

# چند Gap واقعی
df.loc[50:60, "temperature"] = np.nan
df.loc[150:170, "pressure"] = np.nan
df.loc[300:320, "flow"] = np.nan

df.to_csv(
    "timeseries_duplicate_index.csv",
    index=False,
)

print(df.shape)