import numpy as np
from sklearn import  linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

fname= sys.argv[1]
df = pd.read_csv(fname)
df= df.dropna()

training_data= pd.DataFrame()
testing_data= pd.DataFrame()

for p, g in df.groupby("Patient ID"):
    if g.shape[0] >= 12:
        training_data= training_data.append(g[g["Interaction Week"]<g.shape[0]])
        testing_data= testing_data.append(g[g["Interaction Week"]==g.shape[0]])

y_train= training_data["Facit Fatigue Assessment Score"]
y_test= testing_data["Facit Fatigue Assessment Score"]
x_train= training_data.drop(["Facit Fatigue Assessment Score", "Patient ID", "Start", "End"], axis=1)
x_test= testing_data.drop(["Facit Fatigue Assessment Score", "Patient ID", "Start", "End"], axis=1)
# x_train= training_data["schedule_time"].values.reshape(-1,1)
# x_test= testing_data["schedule_time"].values. reshape(-1,1)

#random assignment of training and testing data
# y= df["Facit Fatigue Assessment Score"]
# x= df.drop(["Facit Fatigue Assessment Score", "Patient ID", "Start", "End"], axis=1)
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
regressionScore = regr.score(x_train,y_train)
print("Regression score: ", round(regressionScore,2)) 

y_pred=regr.predict(x_test)
Accuracy=r2_score(y_test,y_pred)
Accuracy= round(Accuracy,2)

print(" R2 Score %.2f" %Accuracy)

plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
sns.regplot(x=y_test,y=y_pred,ci=None,color ='red')

# plt.show()