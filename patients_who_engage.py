import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

df= pd.read_csv('out/PredictedDisengagementWithConfusionMatrix.csv')  

new_df= pd.DataFrame()
for p, g in df.groupby("Patient ID"):
    if g['Active next week'].value_counts()[0]<5:
        new_df= new_df.append(g)

five_patients= new_df["Patient ID"].unique()[:3]
five_df= pd.DataFrame()
for p, g in new_df.groupby("Patient ID"):
    if p in five_patients:
        five_df= five_df.append(g)

five_df= five_df.reset_index(drop=True)
print(len(five_df["Patient ID"].unique()))
print(five_df["Patient ID"].unique())

for p, g in five_df.groupby("Patient ID"):
    # fig= px.line(g, x="Interaction Week", y= "Active Days")
    # fig.add_scatter(x=g["Interaction Week"], y=g["schedule_time"], name="schedule_time")
    # fig.show()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=g["Interaction Week"], y=g['Active Days'], name="Active Days"
        ))
    for c in sorted([c for c in five_df.columns if '_freq' in c]):
    # for c in ["schedule_freq", "medication_freq"]:
        fig.add_trace(
            go.Scatter(
                x=g["Interaction Week"], y=g[c], name=c
            ))

    fig.update_layout(
    title="App feature and number of sessions per week for patients",
    xaxis_title="Interaction Week",
    yaxis_title="Count",
    )

    fig.show()