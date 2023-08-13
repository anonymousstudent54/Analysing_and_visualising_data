import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

df = pd.read_csv('out/PredictedDisengagementWithConfusionMatrix.csv')
df["Active Days"]= df["Active Days"]-1

Engaged= df[(df["Confusion Matrix"]== "True Pos") | (df["Confusion Matrix"]== "False Neg")]
Disengaged= df[(df["Confusion Matrix"]== "True Neg") | (df["Confusion Matrix"]== "False Pos") ]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Patients who engaged in following week", "Patients who disengaged in following week"))

fig.add_trace(go.Histogram(x=Engaged["Active Days"]),
              row=1, col=1)

fig.add_trace(go.Histogram(x=Disengaged["Active Days"]),
              row=1, col=2)

fig.update_layout(
                  title_text="Number of active days using the app in a week for patients who engaged and disengaged in following week")
fig.update_yaxes(range=[0,2000], title="Number of patients")
fig.update_xaxes(title="Number of active days", tickmode='linear')
fig.write_image("figures/active_engaged.png")
fig.show()