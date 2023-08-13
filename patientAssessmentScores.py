import pandas as pd
import numpy as np
import sys


# code from https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
# Find assessment date closest to session time
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

scoresFname= sys.argv[1]
assessment_df= pd.read_csv(scoresFname, parse_dates=['When Completed'])
assessment_df= assessment_df[(assessment_df['Assessment Type']=='CompletedFacitFatigueAssessment') ]

sessionsFname= sys.argv[2]
sessions_df= pd.read_csv(sessionsFname, parse_dates=['Start', 'End'])

for i, s in sessions_df.iterrows():
    patientAssessments= assessment_df[(assessment_df['Patient ID']==s['Patient ID'])].reset_index(drop=True)
    if not patientAssessments.empty:
        nearestIndex= patientAssessments.loc[patientAssessments['When Completed'] == nearest(patientAssessments['When Completed'],s['End'])].index[0]
        sessions_df.at[i,'Facit Fatigue Assessment Score'] = patientAssessments.iloc[nearestIndex]['Primary Score']

sessions_df.to_csv('out/combined_weekly_features_with_scores.csv')