"""
03_eda.py
---------
Exploratory Data Analysis on the final accident dataset.
Generates summary statistics and saves chart images to charts/.

Input : data/final_accident_data_for_powerbi.csv
Output: charts/*.png + printed summary stats
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_PATH = "data/final_accident_data_for_powerbi.csv"
CHART_DIR = "charts"

plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"


def summary_stats(df):
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total accidents      : {len(df):,}")
    print(f"Fatal accidents      : {(df['Severity']=='Fatal').sum():,}")
    print(f"Serious accidents    : {(df['Severity']=='Serious').sum():,}")
    print(f"Minor accidents      : {(df['Severity']=='Minor').sum():,}")
    print(f"Total fatalities     : {df['Fatalities'].sum():,}")
    print(f"Total serious injury : {df['Serious_Injuries'].sum():,}")
    print(f"Total minor injury   : {df['Minor_Injuries'].sum():,}")
    print(f"Avg driver age       : {df['Driver_Age'].mean():.1f}")
    print("=" * 60)


def plot_monthly_trend(df):
    monthly = df.groupby(df["Date"].str[:7]).size()
    monthly.plot(kind="line", marker="o", color="#c0392b")
    plt.title("Monthly Accident Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/monthly_trend.png", dpi=150)
    plt.close()


def plot_state_wise(df):
    state_counts = df["State"].value_counts().sort_values()
    state_counts.plot(kind="barh", color="#2980b9")
    plt.title("State-wise Accident Count")
    plt.xlabel("Number of Accidents")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/state_wise.png", dpi=150)
    plt.close()


def plot_vehicle_type(df):
    df["Vehicle_Type"].value_counts().plot(kind="bar", color="#8e44ad")
    plt.title("Accidents by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/vehicle_type.png", dpi=150)
    plt.close()


def plot_weather_impact(df):
    weather_severity = pd.crosstab(df["Weather_Condition"], df["Severity"])
    weather_severity.plot(kind="bar", stacked=True,
                           color=["#f1c40f", "#e67e22", "#c0392b"])
    plt.title("Weather Impact on Accident Severity")
    plt.xlabel("Weather Condition")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Severity")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/weather_impact.png", dpi=150)
    plt.close()


def plot_time_of_day(df):
    order = ["Morning", "Afternoon", "Evening", "Night"]
    df["Time_of_Day"].value_counts().reindex(order).plot(
        kind="bar", color="#16a085"
    )
    plt.title("Accidents by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/time_of_day.png", dpi=150)
    plt.close()


def plot_severity_pie(df):
    df["Severity"].value_counts().plot(
        kind="pie", autopct="%1.1f%%",
        colors=["#f1c40f", "#e67e22", "#c0392b"]
    )
    plt.title("Accident Severity Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/severity_distribution.png", dpi=150)
    plt.close()


def plot_age_group(df):
    order = ["Under 18", "18-24", "25-34", "35-49", "50-64", "65+"]
    df["Age_Group"].value_counts().reindex(order).plot(kind="bar", color="#34495e")
    plt.title("Accidents by Driver Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/age_group.png", dpi=150)
    plt.close()


def main():
    df = pd.read_csv(INPUT_PATH)
    summary_stats(df)

    plot_monthly_trend(df)
    plot_state_wise(df)
    plot_vehicle_type(df)
    plot_weather_impact(df)
    plot_time_of_day(df)
    plot_severity_pie(df)
    plot_age_group(df)

    print(f"\nAll charts saved to '{CHART_DIR}/' folder.")


if __name__ == "__main__":
    main()
