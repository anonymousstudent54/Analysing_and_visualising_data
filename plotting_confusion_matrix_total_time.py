import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

df = pd.read_csv('out/PredictedDisengagementWithConfusionMatrix.csv')

FP= df[df["Confusion Matrix"]== "False Pos"]
TP= df[df["Confusion Matrix"]== "True Pos"]
FN= df[df["Confusion Matrix"]== "False Neg"]
TN= df[df["Confusion Matrix"]== "True Neg"]

Engaged= df[(df["Confusion Matrix"]== "True Pos") | (df["Confusion Matrix"]== "False Neg")]
Disengaged= df[(df["Confusion Matrix"]== "False Pos") | (df["Confusion Matrix"]== "True Neg")]
print(Engaged["Total Interaction Time (hours)"].max)
bin_size= math.sqrt(len(Engaged.index))
print(bin_size)

# plt.hist(TN["Total Interaction Time (hours)"],  label="True negative", bins=100, range=[0,3] )
# plt.hist(FP["Total Interaction Time (hours)"], label="False positive", bins=100, range=[0,3] )
# plt.ylabel('Number of patients')
# plt.xlabel('Total Interaction Time (hours)')
# plt.legend(loc='upper right')
# plt.title("Total Interaction Time (hours) for patients who DID NOT use app in following week")
# plt.savefig('figures/Totaltime_DIDNOT_use.png')
# plt.show()
# plt.close()


# plt.hist(TP["Total Interaction Time (hours)"],  label="True positive", bins=100, range=[0,10] )
# plt.hist(FN["Total Interaction Time (hours)"], label="False negative", bins=100, range=[0,10] )
# plt.ylabel('Number of patients')
# plt.xlabel('Total Interaction Time (hours)')
# plt.legend(loc='upper right')
# plt.title("Total Interaction Time (hours) for patients who DID use app in following week")
# plt.savefig('figures/Totaltime_DID_use.png')
# plt.show()
# plt.close()



fig = make_subplots(
    rows=1, cols=2, shared_yaxes=True,
    subplot_titles=("Total Interaction Time for patients who DID use app in following week", "Total Interaction Time for patients who DID NOT use app in following week"))

fig.add_trace(go.Histogram(x=TP["Total Interaction Time (hours)"], name="True Positive"),
              row=1, col=1)
fig.add_trace(go.Histogram(x=FN["Total Interaction Time (hours)"], name="False Negative"),
              row=1, col=1)


fig.add_trace(go.Histogram(x=TN["Total Interaction Time (hours)"], name="True Negative"),
              row=1, col=2)
fig.add_trace(go.Histogram(x=FP["Total Interaction Time (hours)"], name="False Positive"),
              row=1, col=2)


# fig.update_layout(
#                   title_text="Total Interaction Time (hours) for confusion matrix quadrants in disengagement classifier ")
# fig.update_yaxes(range=[0,2000])
fig.update_xaxes(title= "Total interaction time (hours)")
fig.write_image("figures/Totaltime_small_multiples.png")
fig.show()







# plt.hist(TN["Total Interaction Time (hours)"],  label="True negative" )
# plt.hist(TP["Total Interaction Time (hours)"], label="True positive" )
# plt.ylabel('Number of patients')
# plt.xlabel('Total Interaction Time (hours)')
# plt.legend(loc='upper right')
# plt.title("Total Interaction Time (hours) for true positive and true negative in disengagement classifier ")
# plt.savefig('figures/TotaltimeTrue_quadrants_LogisticRegression.png')
# plt.show()
# plt.close()

# plt.hist(FN["Total Interaction Time (hours)"], alpha=0.5,label="False negative" )
# plt.hist(FP["Total Interaction Time (hours)"], alpha=0.5, label="False positive" )
# plt.hist(TN["Total Interaction Time (hours)"], alpha=0.5, label="True negative" )
# plt.hist(TP["Total Interaction Time (hours)"], alpha=0.5, label="True positive" )
# plt.ylabel('Number of patients')
# plt.xlabel('Total Interaction Time (hours)')
# plt.legend(loc='upper right')
# plt.title("Total Interaction Time (hours) for confusion matrix quadrants in disengagement classifier ")
# plt.savefig('figures/TotaltimeAll_quadrants_LogisticRegression.png')
# plt.show()
# plt.close()

# plt.hist(FP["Total Interaction Time (hours)"])
# plt.ylabel('Number of patients')
# plt.xlabel('Total Interaction Time (hours)')
# plt.title("Total Interaction Time (hours) for False Postive in disengagement classifier ")
# plt.savefig('figures/TotaltimeFP_LogisticRegression.png')
# plt.show()
# plt.close()

# plt.hist(TN["Total Interaction Time (hours)"])
# plt.ylabel('Number of patients')
# plt.xlabel('Total Interaction Time (hours)')
# plt.title("Total Interaction Time (hours) for True Negative in disengagement classifier ")
# plt.savefig('figures/TotaltimeTN_LogisticRegression.png')
# plt.show()
# plt.close()

# plt.hist(TP["Total Interaction Time (hours)"])
# plt.ylabel('Number of patients')
# plt.xlabel('Total Interaction Time (hours)')
# plt.title("Total Interaction Time (hours) for True Postive in disengagement classifier ")
# plt.savefig('figures/TotaltimeTP_LogisticRegression.png')
# plt.show()
# plt.close()

# plt.hist(FN["Total Interaction Time (hours)"])
# plt.ylabel('Number of patients')
# plt.xlabel('Total Interaction Time (hours)')
# plt.title("Total Interaction Time (hours) for False Negative in disengagement classifier ")
# plt.savefig('figures/TotaltimeFN_LogisticRegression.png')
# plt.show()
# plt.close()


# fig = make_subplots(
#     rows=2, cols=2,
#     subplot_titles=("False Negative", "False Positive", "True Negative", "True Positive"))

# fig.add_trace(go.Histogram(x=FN["Total Interaction Time (hours)"]),
#               row=1, col=1)

# fig.add_trace(go.Histogram(x=FP["Total Interaction Time (hours)"]),
#               row=1, col=2)

# fig.add_trace(go.Histogram(x=TN["Total Interaction Time (hours)"]),
#               row=2, col=1)

# fig.add_trace(go.Histogram(x=TP["Total Interaction Time (hours)"]),
#               row=2, col=2)

# fig.update_layout(height=500, width=700,
#                   title_text="Total Interaction Time (hours) for confusion matrix quadrants in disengagement classifier ")
# fig.update_yaxes(range=[0,2000])
# fig.write_image("figures/Totaltimeall_quadrants_together.png")
# fig.show()


