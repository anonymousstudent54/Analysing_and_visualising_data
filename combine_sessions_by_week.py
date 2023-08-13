import pandas as pd
import numpy as np
import sys

def aggregate_user(u, g):
    data = {}
    data["Clinic ID"]= g["Clinic ID"].iloc[0]
    data['Start'] = g['Start'].min()
    data['End'] = g['End'].max()

    data['N Sessions'] = g['Session'].count()
    # data['Total Interaction Time (minutes)'] = g['Duration'].sum()
    data['Total Interaction Time (hours)'] = g['Duration'].sum() / 60

    # active days (unique days when a session was started)
    data['Active Days'] = len(g['Start'].dt.floor('d').unique())

    # frequency between sessions (in days)
    start_deltas = g['Start'].diff() / pd.Timedelta(1, 'day')

    # calculate maximum gap between days (in days?)
    data['Max usage gap (days)'] = np.max(start_deltas)

    data['Average usage gap (days)'] = np.mean(start_deltas)

    data['Average interaction time per session (minutes)']= np.mean(g['Duration'])
    data['Average interaction frequency per week']= data['N Sessions']/7

    data = pd.Series(data)

    time_data = g[time_cols].sum() 
    freq_data = g[freq_cols].sum() 

    data = pd.concat([data, time_data, freq_data])

    return data


fname = 'data/sessions.csv'
sessions_df= pd.read_csv(fname, parse_dates=['Start', 'End'])

n_patients = len(sessions_df['Patient ID'].unique())

time_cols = sorted([c for c in sessions_df.columns if '_time' in c])
freq_cols = sorted([c for c in sessions_df.columns if '_freq' in c])

result = []
n_recent = 0
skipped = []

for p, g in sessions_df.groupby('Patient ID'):

    # add column with age of the session
    first_start = min(g['Start'])
    g['Interaction Week'] = g['Start'].apply(lambda x: int((x- first_start)/ np.timedelta64(1, 'W'))+1)

    for w, groupedSessions in g.groupby('Interaction Week'):
            curr = aggregate_user(p, groupedSessions)
            curr['Interaction Week'] = w
            curr['Patient ID'] = p
            result.append(curr)

df = pd.DataFrame(result)
df = df.set_index(['Patient ID', 'Interaction Week'])

out_fname = 'out/sessions_combined_by_weeks.csv'
df.to_csv(out_fname)

print("Finished")