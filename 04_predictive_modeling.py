"""
04_predictive_modeling.py  (OPTIONAL)
--------------------------------------
Trains a simple classification model to predict accident Severity
based on accident conditions. This is optional bonus content that
demonstrates the "AI" side of an AI & Data Science project.

Input : data/final_accident_data_for_powerbi.csv
Output: printed model accuracy + feature importance chart
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

INPUT_PATH = "data/final_accident_data_for_powerbi.csv"

FEATURES = [
    "Vehicle_Type", "Weather_Condition", "Road_Type",
    "Road_Surface_Condition", "Light_Condition", "Speed_Limit",
    "Driver_Age", "Time_of_Day", "Is_Weekend", "Season",
    "Number_of_Vehicles_Involved"
]
TARGET = "Severity"


def main():
    df = pd.read_csv(INPUT_PATH)
    data = df[FEATURES + [TARGET]].copy()

    # Encode categorical columns
    encoders = {}
    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, preds), 3))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # Feature importance chart
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    importances.plot(kind="barh", color="#27ae60", figsize=(8, 6))
    plt.title("Feature Importance for Predicting Accident Severity")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("charts/feature_importance.png", dpi=150)
    plt.close()
    print("\nFeature importance chart saved -> charts/feature_importance.png")


if __name__ == "__main__":
    main()
