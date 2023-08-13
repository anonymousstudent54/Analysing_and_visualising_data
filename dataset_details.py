import pandas as pd

df = pd.read_csv('data/sessions.csv', parse_dates=["Start", "End"])

print("Number of patients", len(df["Patient ID"].unique()))
print("Number of clinics", len(df["Clinic ID"].unique()))
print("Start", df["Start"].min())
print("End", df["End"].max())
