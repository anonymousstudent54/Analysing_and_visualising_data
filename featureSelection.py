from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot 
import pandas as pd
import sys

# Code from https://machinelearningmastery.com/feature-selection-for-regression-data/
 
# feature selection
def select_features(X_train, y_train, X_test):
 # configure to select all features
 fs = SelectKBest(score_func=f_regression, k='all')
 # learn relationship from training data
 fs.fit(X_train, y_train)
 # transform train input data
 X_train_fs = fs.transform(X_train)
 # transform test input data
 X_test_fs = fs.transform(X_test)
 return X_train_fs, X_test_fs, fs

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
x_train= training_data.drop(["Facit Fatigue Assessment Score", "Patient ID", "Start", "End", "Unnamed: 0"], axis=1)
x_test= testing_data.drop(["Facit Fatigue Assessment Score", "Patient ID", "Start", "End", "Unnamed: 0"], axis=1)

#random assignment of training and testing data
# y= df["Facit Fatigue Assessment Score"]
# x= df.drop(["Facit Fatigue Assessment Score", "Patient ID", "Start", "End"], axis=1)
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

def addlabels(x,y):
    for i in range(len(x)):
        pyplot.text(i, y[i], y[i], ha = 'center')

X_train_fs, X_test_fs, fs = select_features(x_train, y_train, x_test)
columns= x_train.columns
for i in range(len(fs.scores_)):
 print((columns[i], fs.scores_[i]))
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.xticks(range(len(columns)), columns, rotation='vertical')
# pyplot.xlabel("Feature")
# pyplot.ylabel("Correlation score")
# pyplot.title("Feature correlation score against Fatigue Assessment Score")
# pyplot.savefig('out/Feature Selection.png', bbox_inches="tight")
# pyplot.show()
# pyplot.close()


# plt.show()



# pyplot.scatter(df["medication_freq"],df["Facit Fatigue Assessment Score"])
# pyplot.xlabel('medication_freq')
# pyplot.ylabel('Fatigue Assessment Score')
# pyplot.title("Relationship between frequency of use of medication related app featured and fatigue assessment score")
# pyplot.savefig('out/medication_freq.png', bbox_inches="tight")
# # sns.regplot(x=y_test,y=y_pred,ci=None,color ='red')
# pyplot.show
# pyplot.close

# pyplot.scatter(df["schedule_freq"],df["Facit Fatigue Assessment Score"])
# pyplot.xlabel('schedule_freq')
# pyplot.ylabel('Fatigue Assessment Score')
# pyplot.title("Relationship between frequency of use of schedule related app featured and fatigue assessment score")
# pyplot.savefig('out/schedule_freq.png', bbox_inches="tight")
# # sns.regplot(x=y_test,y=y_pred,ci=None,color ='red')
# pyplot.show
# pyplot.close