import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

df = pd.read_csv("insurance_claims.csv")

# Sort the data based on the Incident Data
df = df.sort_values(by="incident_date").reset_index(drop=True)

# Variable Selection
df = df[
    [
        "incident_date",
        "months_as_customer",
        "age",
        "policy_deductable",
        "policy_annual_premium",
        "umbrella_limit",
        "insured_sex",
        "insured_relationship",
        "capital-gains",
        "capital-loss",
        "incident_type",
        "collision_type",
        "total_claim_amount",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
        "incident_severity",
        "fraud_reported",
    ]
]

# Data Cleaning and One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=[
        "insured_sex",
        "insured_relationship",
        "incident_type",
        "collision_type",
        "incident_severity",
    ],
    drop_first=True,
)

df["fraud_reported"] = df["fraud_reported"].apply(lambda x: 1 if x == "Y" else 0)

df = df.rename(columns={"incident_date": "timestamp", "fraud_reported": "target"})

for i in df.select_dtypes("number").columns:
    df[i] = df[i].apply(float)

data = df[df["timestamp"] < "2015-02-20"].copy()
val = df[df["timestamp"] >= "2015-02-20"].copy()


# evidently to monitor the data and the model performance
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(current_data=val, reference_data=data, column_mapping=None)
data_drift_report.show(mode='inline')

