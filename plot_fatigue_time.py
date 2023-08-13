import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


df= pd.read_csv('data/patient_assessment.csv', parse_dates=['When Completed'])  

# new_df= pd.DataFrame()
# for p, g in df.groupby("Patient ID"):
#     if g['Active next week'].value_counts()[0]<3:
#         new_df= new_df.append(g)

five_patients= df["Patient ID"].unique()[5]
# five_df=df[df["Patient ID"]==patients]
five_df= pd.DataFrame()
for p, g in df.groupby("Patient ID"):
    if p in five_patients:
        five_df= five_df.append(g)

# five_df= five_df.reset_index(drop=True)
# print(len(five_df["Patient ID"].unique()))
# print(five_df["Patient ID"].unique())

for p, g in five_df.groupby("Patient ID"):
    fig = go.Figure()
    for a, group in g.groupby("Assessment Type"):
        fig.add_trace(
            go.Scatter(
                x=group["When Completed"], y=g['Primary Score'], name=a
            ))

    fig.update_layout(
    title="Patient Assessment Scores- Patient "+p,
    # xaxis_title="Interaction Week",
    yaxis_title="Score",
    )

    fig.show()