"""
01_data_cleaning.py
--------------------
Cleans the raw road accident dataset using Pandas:
    - Handles missing values
    - Removes duplicate records
    - Fixes inconsistent text formatting
    - Fixes invalid / impossible values
    - Correct dtypes

Input : data/raw_accident_data.csv
Output: data/cleaned_accident_data.csv
"""

import pandas as pd
import numpy as np

INPUT_PATH = "data/raw_accident_data.csv"
OUTPUT_PATH = "data/cleaned_accident_data.csv"


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_text_columns(df):
    text_cols = [
        "State", "City", "Vehicle_Type", "Weather_Condition",
        "Road_Type", "Road_Surface_Condition", "Light_Condition",
        "Severity", "Driver_Gender"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace({"Nan": np.nan})
            )
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["Accident_ID"], keep="first")
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows")
    return df


def handle_missing_values(df):
    print("\nMissing values before cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Categorical -> fill with mode (most frequent category)
    categorical_cols = ["Weather_Condition", "Road_Surface_Condition", "City"]
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)

    # Numeric -> fill with median
    if "Driver_Age" in df.columns:
        # First flag invalid ages as NaN, then impute
        df.loc[(df["Driver_Age"] < 15) | (df["Driver_Age"] > 100), "Driver_Age"] = np.nan
        median_age = df["Driver_Age"].median()
        df["Driver_Age"] = df["Driver_Age"].fillna(median_age)
        df["Driver_Age"] = df["Driver_Age"].astype(int)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() else "None remaining")
    return df


def fix_dtypes(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Fatalities"] = df["Fatalities"].astype(int)
    df["Serious_Injuries"] = df["Serious_Injuries"].astype(int)
    df["Minor_Injuries"] = df["Minor_Injuries"].astype(int)
    df["Number_of_Vehicles_Involved"] = df["Number_of_Vehicles_Involved"].astype(int)
    df["Speed_Limit"] = df["Speed_Limit"].astype(int)
    return df


def add_total_casualties(df):
    df["Total_Casualties"] = (
        df["Fatalities"] + df["Serious_Injuries"] + df["Minor_Injuries"]
    )
    return df


def main():
    df = load_data(INPUT_PATH)
    df = clean_text_columns(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = fix_dtypes(df)
    df = add_total_casualties(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCleaned dataset saved -> {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
