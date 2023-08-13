from sklearn import  linear_model
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

lut_fname = "data/function_lut.csv"
lut_df = pd.read_csv(lut_fname)
features= lut_df["group"].unique()

print(features)

df = pd.read_csv('out/combined_weekly_features.csv', parse_dates=['Start', 'End'])
df= df.dropna()

# data= pd.DataFrame()
# for p, g in df.groupby("Patient ID"):
#     # if g.shape[0] >= 12:
#     if g["Interaction Week"].max() >= 12:
#         data= data.append(g)

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

print(new_df['Active next week'].value_counts()[0])
print(new_df['Active next week'].value_counts()[1])

for name in features:
    colNames= [c for c in new_df.columns if name in c]
    x= new_df[colNames]
    if x.shape[1]==2:
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
        # Weighting outputs based on inbalanced dataset
        weights = {0:76, 1:24} 
        regr = linear_model.LogisticRegression(class_weight = weights, max_iter=10000)
        regr.fit(x_train, y_train)
        training_accuracy = regr.score(x_train,y_train)*100
        y_pred=regr.predict(x_test)
        Accuracy=f1_score(y_test,y_pred)*100
        print(name, round(training_accuracy,2), round(Accuracy,2))

