import pandas as pd
import numpy as np
import sys
from time import sleep
from tqdm import tqdm

# code from https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
# Find assessment date closest to session time
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

scoresFname= sys.argv[1]
assessment_df= pd.read_csv(scoresFname, parse_dates=['When Completed'])
sessionsFname= sys.argv[2]
sessions_df= pd.read_csv(sessionsFname, parse_dates=['Start', 'End'])

assessments= assessment_df["Assessment Type"].unique()

print(len(assessment_df.index))

for a in assessments:
    print("\item ", a)
# print(assessments)

for assessmentType in assessments:
    assessment= []
    assessment_df= assessment_df[(assessment_df['Assessment Type']==assessmentType)]
    print(assessmentType)
    for s in tqdm(sessions_df.itertuples()):
        patientID= s[1]
        patientAssessment= assessment_df[(assessment_df['Patient ID']==patientID)].reset_index(drop=True)
        if not patientAssessment.empty:
            nearestIndex= patientAssessment.loc[patientAssessment['When Completed'] == nearest(patientAssessment['When Completed'],s[4])].index[0]
            assessment.append(patientAssessment.iloc[nearestIndex]['Primary Score'])
        else:
            assessment.append(float("NaN"))
    sessions_df=sessions_df.append(assessment)

sessions_df.to_csv('out/sessions_with_ALL_scores_by_feature.csv')
