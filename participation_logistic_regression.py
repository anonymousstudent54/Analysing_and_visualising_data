from sklearn import  linear_model
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std

df = pd.read_csv('out/combined_weekly_features.csv')
df= df.dropna()

# Create column with binary value indicating 1 if patient will use app in following week and 0 otherwise
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

y= new_df["Active next week"]
x= new_df.drop(["Active next week", "Patient ID", "Start", "End", "Interaction Week", "Clinic ID"], axis=1)

# print(new_df['Active next week'].value_counts()[0])
# print(new_df['Active next week'].value_counts()[1])
weights = {0:76, 1:24}
regr = linear_model.LogisticRegression(class_weight = weights, max_iter=10000)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(regr, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

y_pred = cross_val_predict(regr, x, y, cv=3)
Accuracy=f1_score(y,y_pred)*100
print("Accuracy of the model is %.2f" %Accuracy)

new_df["Y_pred"]= y_pred

def confusionMatrix(x):
    if x["Active next week"]==1 and x["Y_pred"]==1:
        return "True Pos"
    elif x["Active next week"]==1 and x["Y_pred"]==0:
        return "False Neg"
    elif x["Active next week"]==0 and x["Y_pred"]==1:
        return "False Pos"
    else:
        return "True Neg"

new_df["Confusion Matrix"]= new_df.apply(confusionMatrix, axis=1)
new_df= new_df.reset_index(drop=True)
new_df.to_csv('out/PredictedDisengagementWithConfusionMatrix.csv')  

# Calculate feature importance
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
regr.fit(x_train, y_train)
training_accuracy = regr.score(x_train,y_train)
print("training accuracy is: ",training_accuracy *100)
y_pred=regr.predict(x_test)
Accuracy=f1_score(y_test,y_pred)*100
print("Accuracy of the model is %.2f" %Accuracy)

# columns= x.columns
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# #code from https://machinelearningmastery.com/calculate-feature-importance-with-python/
# importance = regr.coef_[0]
# # summarize feature importance
# for i,v in enumerate(importance):
#     print(columns[i], 'Score:',v)
# # plot feature importance
# plt.bar([x for x in range(len(importance))], importance)
# plt.xticks(range(len(columns)), columns, rotation='vertical')
# plt.xlabel("Feature")
# plt.ylabel("Coefficient")
# plt.savefig('figures/Logistic_regression_feature_importance2.png', bbox_inches="tight")
# plt.show()

# # Create confusion matrox
# cm=confusion_matrix(y, y_pred)
# print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=regr.classes_)
# disp.plot()
# plt.savefig('figures/ConfusionMatrix2.png', bbox_inches="tight")
# plt.show()