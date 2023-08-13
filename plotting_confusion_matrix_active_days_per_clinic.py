import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product


df = pd.read_csv('out/PredictedDisengagementWithConfusionMatrix.csv')
df["Active Days"]= df["Active Days"]-1

patient_df = pd.read_csv('data/patients.csv', usecols= ["Clinic ID", "Clinic Name"])

df=pd.merge(df, patient_df, on="Clinic ID")
# df = df.join(patient_df.set_index('Col1'), on='Col3')
# df["Clinic name"]= patient_df[patient_df["Clinic ID"]==df["Clinic ID"]]["Clinic Name"].iloc[0]

FP= df[df["Confusion Matrix"]== "False Pos"]
TP= df[df["Confusion Matrix"]== "True Pos"]
FN= df[df["Confusion Matrix"]== "False Neg"]
TN= df[df["Confusion Matrix"]== "True Neg"]

Engaged= df[(df["Confusion Matrix"]== "True Pos") | (df["Confusion Matrix"]== "False Neg")]
Disengaged= df[(df["Confusion Matrix"]== "True Neg") | (df["Confusion Matrix"]== "False Pos") ]

clinics= df["Clinic Name"].unique()

fig = make_subplots( rows=5, cols=5,
    subplot_titles=clinics)

lst = [6,6]
result = list(product(*(range(1,l) for l in lst)))

for i in range(len(clinics)):
    clinic_engaged= Engaged[Engaged["Clinic Name"]==clinics[i]]
    fig.add_trace(go.Histogram(x=clinic_engaged["Active Days"]), row=result[i][0], col=result[i][1])

fig.update_layout(title_text="Active days by clinic for patients who engaged with app in following week", showlegend=False)
fig.update_annotations(font_size=10)
# fig.update_yaxes(range=[0,2000])
# fig.tight_layout(pad=5.0)
fig.write_image("figures/clinics/clinic_engagement_active_days.png")
fig.show()