import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
import math


df= pd.read_csv('out/combined_weekly_features_with_scores.csv')  
print(df.shape)
patient_df = pd.read_csv('data/patients.csv', usecols= ["Clinic ID", "Clinic Name"])
patient_df=patient_df.drop_duplicates()

df=pd.merge(df, patient_df[["Clinic ID", "Clinic Name"]], on="Clinic ID")
print(df.shape)

columns= df.columns

improvement= []
for p, g in df.groupby("Patient ID"):
    new_patient={}
    new_patient["Patient ID"]=p
    new_patient["Clinic Name"]= g["Clinic Name"].iloc[0]
    last_week= g["Interaction Week"].max()
    # print("Number of 1s", g[g["Interaction Week"]==1].shape)
    score_week1= g[g["Interaction Week"]==1]["Facit Fatigue Assessment Score"].iloc[0]
    score_end= g[g["Interaction Week"]==last_week]["Facit Fatigue Assessment Score"].iloc[0]
    # print("score",score_end-score_week1)

    new_patient["Improvement"]=score_end-score_week1

    # new_patient["Total time"]=g[g["Total Interaction Time (hours)"]].sum()
    # new_patient["N Sessions"]=g[g["N Sessions"]].sum()
    # new_patient["Active Days"]=g[g["Active Days"]].sum()
    # for c in sorted([c for c in df.columns if '_freq' in c]):
    #     new_patient[c]=g[g[c]].sum()
    improvement.append(new_patient)

improvement_df = pd.DataFrame(improvement)
print(improvement_df.shape)
improvement_df=improvement_df.dropna()
print(improvement_df.shape)

# print(improvement_df.columns)
clinics= improvement_df["Clinic Name"].unique()[:4]
print(len(clinics))

# bin_size= round(math.sqrt(len(improvement_df.index)/4))
# print(bin_size)



fig = make_subplots( rows=2, cols=2,
    subplot_titles=clinics)

to_plot= improvement_df[improvement_df["Clinic Name"]==clinics[0]]
bin_size= round(math.sqrt(len(to_plot.index)))
print(bin_size)
fig.add_trace(go.Histogram(x=to_plot['Improvement'], nbinsx=15, name=clinics[0]), row=1, col=1)

to_plot= improvement_df[improvement_df["Clinic Name"]==clinics[1]]
bin_size= round(math.sqrt(len(to_plot.index)))
fig.add_trace(go.Histogram(x=to_plot['Improvement'], nbinsx=15, name=clinics[1]), row=1, col=2)

to_plot= improvement_df[improvement_df["Clinic Name"]==clinics[2]]
bin_size= round(math.sqrt(len(to_plot.index)))
fig.add_trace(go.Histogram(x=to_plot['Improvement'], nbinsx=bin_size, name=clinics[2]), row=2, col=1)

to_plot= improvement_df[improvement_df["Clinic Name"]==clinics[3]]
bin_size= round(math.sqrt(len(to_plot.index)))
fig.add_trace(go.Histogram(x=to_plot['Improvement'], nbinsx=15, name=clinics[3]), row=2, col=2)

fig.update_xaxes(title_text="Improvement in fatigue score",range=[-30,50], row=1, col=1)
fig.update_xaxes(title_text="Improvement in fatigue score",range=[-30,50],  row=1, col=2)
fig.update_xaxes(title_text="Improvement in fatigue score",range=[-30,50],  row=2, col=1)
fig.update_xaxes(title_text="Improvement in fatigue score",range=[-30,50],  row=2, col=2)

fig.update_yaxes(title_text="Count", range=[0, 140], row=1, col=1)
fig.update_yaxes(title_text="Count", range=[0, 140], row=1, col=2)
fig.update_yaxes(title_text="Count", range=[0, 140], row=2, col=1)
fig.update_yaxes(title_text="Count", range=[0, 140], row=2, col=2)


fig.update_layout(
                  title_text="Histogram of improvement in patient fatigue scores in clinics")

# fig.update_layout(
# title="Histogram of improvement in patient fatigue scores in clinincs",
# # xaxis_title="Improvement in fatigue score",
# # yaxis_title="Count",
# )

fig.show()