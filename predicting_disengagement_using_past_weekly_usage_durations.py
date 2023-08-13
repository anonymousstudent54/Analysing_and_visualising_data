import pandas as pd
import sys
from pathlib import Path
from sklearn import  linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import sys

# fname= sys.argv[1]
# df = pd.read_csv(fname)
df = pd.read_csv('out/combined_weekly_features.csv')
print(len(df.index))
df= df.dropna()

results_df= []
for p, g in df.groupby("Patient ID"):
    interactionWeeks= g["Interaction Week"].max()
    # if g.shape[0] >= 12:
    if interactionWeeks >=12:
        interactionWeeks= g["Interaction Week"].max()
        patient = {}
        patient['Patient ID']= p
        patient["Interaction duration in weeks"]= interactionWeeks
        for x in range(1,13):
            if not g[g["Interaction Week"]==x].empty:
                patient["Week %s interaction" % x]= g[g["Interaction Week"]==x]["Total Interaction Time (hours)"].values[0]
            else: 
                patient["Week %s interaction" % x]= 0
        results_df.append(patient)
results_df= pd.DataFrame(results_df)

Y= (results_df["Week 12 interaction"]>0).astype(int)
print(Y.value_counts()[0])
print(Y.value_counts()[1])
X= results_df.drop(["Week 12 interaction", "Interaction duration in weeks", "Patient ID"], axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

regr = linear_model.LogisticRegression( max_iter=10000)
regr.fit(x_train, y_train)

training_accuracy = regr.score(x_train,y_train)
print("training accuracy is: ",training_accuracy *100) 

y_pred=regr.predict(x_test)
print(y_test)
print(y_pred)
Accuracy=f1_score(y_test,y_pred)*100

print("Accuracy of the model is %.2f" %Accuracy)