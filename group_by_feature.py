from sklearn import  linear_model
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from pathlib import Path

# df = pd.read_csv('out/sessions_combined_by_weeks.csv')
# df= df.dropna()

# fname = sys.argv[1]
fname = 'out/sessions_combined_by_weeks.csv'
sessions_df = pd.read_csv(fname, parse_dates=['Start', 'End'])
# sessions_df = sessions_df.drop(columns='Unnamed: 0')

lut_fname = 'data/function_lut.csv'
lut_df = pd.read_csv(lut_fname)
lut_df = lut_df.groupby('group')['name'].apply(list)

n_patients = len(sessions_df['Patient ID'].unique())

# apply LUT to df
main_features = list(lut_df.index)
to_exclude = ['general']

# group (and aggregate) columns based on the LUT
for suffix in ['_time', '_freq']:
    for group, items in lut_df.iteritems():
        # print(items)
        cols = []
        for i in items:
            cols += [
                col for col in sessions_df.columns 
                if col.startswith(i) and col.endswith(suffix)
                ]

        if group not in to_exclude:
            sessions_df[group + suffix] = sessions_df[cols].sum(axis=1)
        sessions_df = sessions_df.drop(columns=cols)

time_cols = sorted([c for c in sessions_df.columns if '_time' in c])
freq_cols = sorted([c for c in sessions_df.columns if '_freq' in c])

sessions_df = sessions_df.set_index(['Patient ID', 'Interaction Week'])

out_fname = 'combined_weekly_features.csv'
sessions_df.to_csv(Path('out') / out_fname)
print('Finished')