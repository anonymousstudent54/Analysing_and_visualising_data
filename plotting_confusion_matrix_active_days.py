import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

df = pd.read_csv('out/PredictedDisengagementWithConfusionMatrix.csv')
df["Active Days"]= df["Active Days"]-1

FP= df[df["Confusion Matrix"]== "False Pos"]
TP= df[df["Confusion Matrix"]== "True Pos"]
FN= df[df["Confusion Matrix"]== "False Neg"]
TN= df[df["Confusion Matrix"]== "True Neg"]


plt.hist(TN["Active Days"],  label="True negative" )
plt.hist(FP["Active Days"], label="False positive" )
plt.ylabel('Number of patients')
plt.xlabel('Active Days')
plt.legend(loc='upper right')
plt.title("Active Days for patients who DID NOT use app in following week")
plt.savefig('figures/ActiveDays_DIDNOT_use.png')
plt.show()
plt.close()


plt.hist(TP["Active Days"],  label="True positive" )
plt.hist(FN["Active Days"], label="False negative" )
plt.ylabel('Number of patients')
plt.xlabel('Active Days')
plt.legend(loc='upper right')
plt.title("Active Days for patients who DID use app in following week")
plt.savefig('figures/ActiveDays_DID_use.png')
plt.show()
plt.close()



plt.hist(TN["Active Days"],  label="True negative" )
plt.hist(TP["Active Days"], label="True positive" )
plt.ylabel('Number of patients')
plt.xlabel('Active Days')
plt.legend(loc='upper right')
plt.title("Active Days for true positive and true negative in disengagement classifier ")
plt.savefig('figures/True_quadrants_LogisticRegression.png')
plt.show()
plt.close()

plt.hist(FN["Active Days"], alpha=0.5,label="False negative" )
plt.hist(FP["Active Days"], alpha=0.5, label="False positive" )
plt.hist(TN["Active Days"], alpha=0.5, label="True negative" )
plt.hist(TP["Active Days"], alpha=0.5, label="True positive" )
plt.ylabel('Number of patients')
plt.xlabel('Active Days')
plt.legend(loc='upper right')
plt.title("Active Days for confusion matrix quadrants in disengagement classifier ")
plt.savefig('figures/All_quadrants_LogisticRegression.png')
plt.show()
plt.close()

plt.hist(FP["Active Days"])
plt.ylabel('Number of patients')
plt.xlabel('Active Days')
plt.title("Active Days for False Postive in disengagement classifier ")
plt.savefig('figures/FP_LogisticRegression.png')
plt.show()
plt.close()

plt.hist(TN["Active Days"])
plt.ylabel('Number of patients')
plt.xlabel('Active Days')
plt.title("Active Days for True Negative in disengagement classifier ")
plt.savefig('figures/TN_LogisticRegression.png')
plt.show()
plt.close()

plt.hist(TP["Active Days"])
plt.ylabel('Number of patients')
plt.xlabel('Active Days')
plt.title("Active Days for True Postive in disengagement classifier ")
plt.savefig('figures/TP_LogisticRegression.png')
plt.show()
plt.close()

plt.hist(FN["Active Days"])
plt.ylabel('Number of patients')
plt.xlabel('Active Days')
plt.title("Active Days for False Negative in disengagement classifier ")
plt.savefig('figures/FN_LogisticRegression.png')
plt.show()
plt.close()


fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("False Negative", "False Positive", "True Negative", "True Positive"))

fig.add_trace(go.Histogram(x=FN["Active Days"]),
              row=1, col=1)

fig.add_trace(go.Histogram(x=FP["Active Days"]),
              row=1, col=2)

fig.add_trace(go.Histogram(x=TN["Active Days"]),
              row=2, col=1)

fig.add_trace(go.Histogram(x=TP["Active Days"]),
              row=2, col=2)

fig.update_layout(height=500, width=700,
                  title_text="Active Days for confusion matrix quadrants in disengagement classifier ")
fig.update_yaxes(range=[0,2000])
fig.write_image("figures/all_quadrants_together.png")
fig.show()


