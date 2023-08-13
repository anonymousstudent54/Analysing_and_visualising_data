import numpy as np
from sklearn import  linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from sklearn.preprocessing import PolynomialFeatures


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
z_train= training_data.drop(["Facit Fatigue Assessment Score", "Patient ID", "Start", "End"], axis=1)
z_test= testing_data.drop(["Facit Fatigue Assessment Score", "Patient ID", "Start", "End"], axis=1)

P = PolynomialFeatures(degree=2, include_bias=False).fit(z_train)
x_train= P.transform(z_train)
x_test = P.transform(z_test)

# random assignment of training and testing data
# y= df["Facit Fatigue Assessment Score"]
# x= df.drop(["Facit Fatigue Assessment Score", "Patient ID", "Start", "End"], axis=1)
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
training_accuracy = regr.score(x_train,y_train)
print("Training accuracy is: ",training_accuracy *100) 

y_pred=regr.predict(x_test)
Accuracy=r2_score(y_test,y_pred)

print("R2 Score of the model is %.2f" %Accuracy)
