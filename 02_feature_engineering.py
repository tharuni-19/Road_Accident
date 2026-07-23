"""
02_feature_engineering.py
--------------------------
Creates derived features from the cleaned dataset, ready to be
loaded straight into Power BI.

New features:
    - Year, Month, Month_Name, Day, Day_Name, Hour
    - Time_of_Day  (Morning / Afternoon / Evening / Night)
    - Is_Weekend
    - Season
    - Age_Group
    - Severity_Score (weighted numeric score, useful for KPI cards)
    - Is_Fatal_Accident (flag, 1/0)

Input : data/cleaned_accident_data.csv
Output: data/final_accident_data_for_powerbi.csv
"""

import pandas as pd

INPUT_PATH = "data/cleaned_accident_data.csv"
OUTPUT_PATH = "data/final_accident_data_for_powerbi.csv"


def time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def season_from_month(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


def age_group(age):
    if age < 18:
        return "Under 18"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 50:
        return "35-49"
    elif age < 65:
        return "50-64"
    else:
        return "65+"


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])

    # Combine Date + Time into a proper datetime for hour extraction
    df["Datetime"] = pd.to_datetime(
        df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str),
        errors="coerce"
    )

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["Day"] = df["Date"].dt.day
    df["Day_Name"] = df["Date"].dt.strftime("%A")
    df["Hour"] = df["Datetime"].dt.hour.fillna(0).astype(int)

    df["Time_of_Day"] = df["Hour"].apply(time_of_day)
    df["Is_Weekend"] = df["Day_Name"].isin(["Saturday", "Sunday"])
    df["Season"] = df["Month"].apply(season_from_month)
    df["Age_Group"] = df["Driver_Age"].apply(age_group)

    # Weighted severity score: useful as a single KPI (higher = worse)
    df["Severity_Score"] = (
        df["Fatalities"] * 5 + df["Serious_Injuries"] * 2 + df["Minor_Injuries"] * 1
    )
    df["Is_Fatal_Accident"] = (df["Severity"] == "Fatal").astype(int)

    df = df.drop(columns=["Datetime"])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature-engineered dataset saved -> {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()
