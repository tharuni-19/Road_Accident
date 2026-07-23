"""
generate_dataset.py
--------------------
Generates a synthetic Road Accident dataset that mimics real-world
accident records (similar in structure to common Kaggle "Road/US
Accidents" datasets).

If you already have a real dataset (e.g. from Kaggle, government open
data portal, etc.), SKIP this script — just place your CSV at
data/raw_accident_data.csv with similar column names (or update the
column names in 01_data_cleaning.py) and continue the pipeline from
there.

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_ROWS = 6000

states = [
    "Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Karnataka", "Delhi",
    "Gujarat", "Rajasthan", "West Bengal", "Madhya Pradesh", "Punjab",
    "Kerala", "Telangana", "Bihar", "Haryana", "Andhra Pradesh"
]

state_coords = {
    "Maharashtra": (19.75, 75.71), "Tamil Nadu": (11.13, 78.66),
    "Uttar Pradesh": (26.85, 80.95), "Karnataka": (15.32, 75.71),
    "Delhi": (28.70, 77.10), "Gujarat": (22.26, 71.19),
    "Rajasthan": (27.02, 74.22), "West Bengal": (22.99, 87.86),
    "Madhya Pradesh": (22.97, 78.66), "Punjab": (31.15, 75.34),
    "Kerala": (10.85, 76.27), "Telangana": (18.11, 79.02),
    "Bihar": (25.10, 85.31), "Haryana": (29.06, 76.09),
    "Andhra Pradesh": (15.91, 79.74)
}

vehicle_types = ["Two-Wheeler", "Car", "Truck", "Bus", "Auto-Rickshaw", "Bicycle", "Pedestrian"]
weather_conditions = ["Clear", "Rainy", "Foggy", "Cloudy", "Stormy", "Hazy"]
road_types = ["Highway", "City Road", "Rural Road", "Expressway", "Bridge/Flyover"]
road_surface = ["Dry", "Wet", "Under Construction", "Icy", "Muddy"]
light_conditions = ["Daylight", "Dark - Lit", "Dark - Unlit", "Dawn/Dusk"]
severity_levels = ["Minor", "Serious", "Fatal"]
genders = ["Male", "Female"]

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days),
                              seconds=random.randint(0, 86399))

start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)

rows = []
for i in range(1, N_ROWS + 1):
    dt = random_date(start_date, end_date)
    state = random.choice(states)
    lat_base, lon_base = state_coords[state]

    severity = np.random.choice(severity_levels, p=[0.55, 0.30, 0.15])

    if severity == "Fatal":
        fatalities = np.random.randint(1, 4)
        serious_injuries = np.random.randint(0, 3)
        minor_injuries = np.random.randint(0, 2)
    elif severity == "Serious":
        fatalities = 0
        serious_injuries = np.random.randint(1, 4)
        minor_injuries = np.random.randint(0, 3)
    else:
        fatalities = 0
        serious_injuries = 0
        minor_injuries = np.random.randint(1, 4)

    row = {
        "Accident_ID": f"ACC{i:05d}",
        "Date": dt.strftime("%Y-%m-%d"),
        "Time": dt.strftime("%H:%M:%S"),
        "State": state,
        "City": f"{state} City {random.randint(1,5)}",
        "Latitude": round(lat_base + np.random.uniform(-1.5, 1.5), 6),
        "Longitude": round(lon_base + np.random.uniform(-1.5, 1.5), 6),
        "Vehicle_Type": random.choice(vehicle_types),
        "Number_of_Vehicles_Involved": np.random.randint(1, 5),
        "Weather_Condition": np.random.choice(weather_conditions, p=[0.45,0.2,0.1,0.15,0.05,0.05]),
        "Road_Type": random.choice(road_types),
        "Road_Surface_Condition": random.choice(road_surface),
        "Light_Condition": random.choice(light_conditions),
        "Speed_Limit": random.choice([30, 40, 50, 60, 80, 100]),
        "Driver_Age": np.random.randint(18, 75),
        "Driver_Gender": random.choice(genders),
        "Severity": severity,
        "Fatalities": fatalities,
        "Serious_Injuries": serious_injuries,
        "Minor_Injuries": minor_injuries,
    }
    rows.append(row)

df = pd.DataFrame(rows)

# --- Intentionally inject real-world messiness for the cleaning step ---
# 1. Missing values
for col in ["Weather_Condition", "Road_Surface_Condition", "Driver_Age", "City"]:
    idx = df.sample(frac=0.03, random_state=1).index
    df.loc[idx, col] = np.nan

# 2. Duplicate rows
dupes = df.sample(frac=0.01, random_state=2)
df = pd.concat([df, dupes], ignore_index=True)

# 3. Inconsistent text casing / whitespace
df.loc[df.sample(frac=0.05, random_state=3).index, "Vehicle_Type"] = \
    df.loc[df.sample(frac=0.05, random_state=3).index, "Vehicle_Type"].str.upper()
df.loc[df.sample(frac=0.03, random_state=4).index, "Weather_Condition"] = \
    df.loc[df.sample(frac=0.03, random_state=4).index, "Weather_Condition"].astype(str) + "  "

# 4. A few invalid ages (data entry errors)
bad_age_idx = df.sample(frac=0.005, random_state=5).index
df.loc[bad_age_idx, "Driver_Age"] = -1

df.to_csv("data/raw_accident_data.csv", index=False)
print(f"Generated {len(df)} rows -> data/raw_accident_data.csv")
