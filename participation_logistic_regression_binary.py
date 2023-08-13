from sklearn import  linear_model
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('out/combined_weekly_features.csv')
df= df.dropna()


new_df= pd.DataFrame()
for p, g in df.groupby("Patient ID"):
    if g["Interaction Week"].max() >= 12:
        for index, row in g.iterrows():
            if not index==len(g)-1:
                if g[g["Interaction Week"]==row["Interaction Week"]+1].empty:
                    row["Active next week"]=0
                    new_df= new_df.append(row)
                else:
                    row["Active next week"]=1
                    new_df= new_df.append(row)

#random assignment of training and testing data
y= new_df["Active next week"]
# x= new_df.drop(["Active next week", "Patient ID", "Start", "End", "Interaction Week", "N Sessions", "Total Interaction Time (hours)", "Active Days", "Max usage gap (days)", "Average usage gap (days)",	"Average interaction time per session (minutes)", "Average interaction frequency per week"], axis=1)
x= new_df.drop(["Active next week", "Patient ID", "Start", "End", "Interaction Week"], axis=1)

for (columnName, columnData) in x.iteritems():
    if '_time' or '_freq' in columnName:
        x[columnName]= (x[columnName]>0).astype(int)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print(new_df['Active next week'].value_counts()[0])
print(new_df['Active next week'].value_counts()[1])
weights = {0:76, 1:24}
regr = linear_model.LogisticRegression(class_weight = weights, max_iter=10000)
regr.fit(x_train, y_train)
training_accuracy = regr.score(x_train,y_train)
print("training accuracy is: ",training_accuracy *100)

y_pred=regr.predict(x_test)
Accuracy=f1_score(y_test,y_pred)*100
print("Accuracy of the model is %.2f" %Accuracy)

#Logistic regression 
# training accuracy is:  76.10239162929746
# Accuracy of the model is 85.34

# weighted
# training accuracy is:  62.322496263079216
# Accuracy of the model is 67.61
