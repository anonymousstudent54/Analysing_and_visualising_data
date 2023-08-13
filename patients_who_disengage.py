import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


df= pd.read_csv('out/PredictedDisengagementWithConfusionMatrix.csv')  

new_df= pd.DataFrame()
for p, g in df.groupby("Patient ID"):
    if g['Active next week'].value_counts()[0]>6:
        new_df= new_df.append(g)

aggregated=[]
for i, g in df.groupby("Interaction Week"):
    new={}
    new["Interaction Week"]=i
    new["Active Days"]= g["Active Days"].mean()
    for c in sorted([c for c in df.columns if '_freq' in c]):
        new[c]= g[c].mean()
    aggregated.append(new)
aggregated_df= pd.DataFrame(aggregated)

patients= new_df["Patient ID"].unique()[1]
five_df=new_df[new_df["Patient ID"]==patients]
# pd.DataFrame()
# for p, g in new_df.groupby("Patient ID"):
#     if p in five_patients:
#         five_df= five_df.append(g)

# five_df= five_df.reset_index(drop=True)
# print(len(five_df["Patient ID"].unique()))
# print(five_df["Patient ID"].unique())



colours=dict(assessment_freq="red",
             complete_freq='yellow',
             diaries_freq='blue',
             goals_freq='green',
             library_freq="pink",
             measurements_freq='purple',
             medication_freq='orange',
             messages_freq='darkseagreen',
             outcomes_freq='greenyellow',
             programme_freq='brown',
             schedule_freq='cyan',
             symptom_freq='coral'
             )
for p, g in five_df.groupby("Patient ID"):
    fig = make_subplots( rows=1, cols=2,
        subplot_titles=("Average of patients", p))
    
    fig.add_trace(
        go.Bar(
            x=aggregated_df["Interaction Week"], y=aggregated_df['Active Days'], name="Active Days", marker=dict(color='cornflowerblue'), showlegend=False
        ), row=1, col=1)
    for c in sorted([c for c in aggregated_df.columns if '_freq' in c]):
    # for c in ["schedule_freq", "medication_freq"]:
        fig.add_trace(
            go.Scatter(
                x=aggregated_df["Interaction Week"], y=aggregated_df[c], name=c, marker=dict(color=colours[c]), showlegend=False
            ), row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=g["Interaction Week"], y=g['Active Days'], name="Active Days", marker=dict(color='cornflowerblue')
        ), row=1, col=2)
    for c in sorted([c for c in five_df.columns if '_freq' in c]):
    # for c in ["schedule_freq", "medication_freq"]:
        fig.add_trace(
            go.Scatter(
                x=g["Interaction Week"], y=g[c], name=c, marker=dict(color=colours[c])
            ), row=1, col=2)

    fig.update_xaxes(title_text="Interaction week", range=[0,35], row=1, col=1)
    fig.update_xaxes(title_text="Interaction week", range=[0,35], row=1, col=2)

    fig.update_layout(
    title="App feature usage and number active days per week for patients",
    )

    fig.show()








# for p, g in five_df.groupby("Patient ID"):
#     # fig= px.line(g, x="Interaction Week", y= "Active Days")
#     # fig.add_scatter(x=g["Interaction Week"], y=g["schedule_time"], name="schedule_time")
#     # fig.show()
#     fig = go.Figure()
#     fig.add_trace(
#         go.Bar(
#             x=g["Interaction Week"], y=g['Active Days'], name="Active Days"
#         ))
#     for c in sorted([c for c in five_df.columns if '_freq' in c]):
#     # for c in ["schedule_freq", "medication_freq"]:
#         fig.add_trace(
#             go.Scatter(
#                 x=g["Interaction Week"], y=g[c], name=c
#             ))

#     fig.update_layout(
#     title="App feature and number of sessions per week for patients",
#     xaxis_title="Interaction Week",
#     yaxis_title="Count",
#     )

#     fig.show()






# for p, g in five_df.groupby("Patient ID"):
#     # fig= px.line(g, x="Interaction Week", y= "Active Days")
#     # fig.add_scatter(x=g["Interaction Week"], y=g["schedule_time"], name="schedule_time")
#     # fig.show()
#     fig = go.Figure()
#     fig.update_layout(
#         yaxis=dict(
#             title="Count", 
#             side="left"
#         ),
#         yaxis2=dict(
#             title="Engagement",
#             overlaying="y",
#             side="right",
#             tickmode = 'array',
#             tickvals = [0,1],)
#     )

#     fig.add_trace(
#         go.Scatter(
#             x=g["Interaction Week"]+1, y=g["Active next week"], name="Active", yaxis="y2", line=dict(color='firebrick', width=3))
#         # go.Bar( x=g["Interaction Week"]+1, y=g["Active next week"], name="Active", yaxis="y2")
#         )
#     for c in sorted([c for c in five_df.columns if '_freq' in c]):
#         fig.add_trace(
#             go.Scatter(
#                 x=g["Interaction Week"], y=g[c], name=c, yaxis="y"
#             ))

#     fig.update_layout(
#     title="App feature and number of sessions per week for patient "+p,
#     xaxis_title="Interaction Week",
#     )

#     fig.show()