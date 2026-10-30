import pandas as pd
from collections import defaultdict
import numpy as np
import json

file_name = 'assistments_2009_2010.csv'
heads = ['order_id', 'user_id', 'problem_id', 'list_skills', 'correct', 'ms_first_response_time']
df = pd.read_csv(file_name, usecols = heads, dtype = {'list_skill_ids': str})
df = df[~df['list_skills'].isnull()]
df['skill_list'] = df['list_skills'].apply(lambda x: x.split(';'))
user_list = np.unique(df['user_id'].tolist())
prob_list = np.unique(df['problem_id'].tolist())
skill_list = df['skill_list'].tolist()
skills = []
for l in skill_list:
    skills.extend(l)
know_list = np.unique(skills)

user_map = dict(zip(user_list, range(1, len(user_list) + 1)))
prob_map = dict(zip(prob_list, range(1, len(prob_list) + 1)))
know_map = dict(zip(know_list, range(1, len(know_list) + 1)))

# with open('know_map.json', 'w') as f:
#     json.dump(know_map, f)

# with open('know_list.json', 'w') as f:
#     json.dump(know_list.tolist(), f)

# asd

df.sort_values(by = 'order_id', ascending = True, inplace = True)

from collections import defaultdict
user_dict = defaultdict(list)
for row in df.itertuples():
    user = getattr(row, 'user_id')
    corr = getattr(row, 'correct') == 1
    knows = [know_map[k] for k in getattr(row, 'skill_list')]
    prob = prob_map[getattr(row, 'problem_id')]
    timestamp = int(getattr(row, 'order_id'))
    try:
        timelag = int(getattr(row, 'ms_first_response_time'))
    except:
        timelag = 0
    user_dict[user].append((prob, knows, corr, timestamp, timelag))

from tqdm import tqdm
import numpy as np
n_knows = len(know_map)

tt = np.zeros((n_knows + 1, n_knows + 1))
tf = np.zeros((n_knows + 1, n_knows + 1))
ft = np.zeros((n_knows + 1, n_knows + 1))
ff = np.zeros((n_knows + 1, n_knows + 1))

seq_list = list(user_dict.values())

for seq in tqdm(seq_list):
    t = np.zeros(n_knows + 1)
    f = np.zeros(n_knows + 1)
    for _, knows, corr, _, _ in seq:
        know = knows[0]
        if corr:
            tt[:, know] += t
            tt[know, :] += t
            ft[:, know] += f
            tf[know, :] += f
        else:
            ff[:, know] += f
            ff[know, :] += f
            tf[:, know] += t
            ft[know, :] += t
        if corr:
            t[know] += 1
        else:
            f[know] += 1

cold_thresh = 5

rel_filt = (tt + ff + tf + ft) >= 4*cold_thresh
pre_filt = (tf + ft) >= 2*cold_thresh

rel_map = (tt + ff)/np.clip(tt + ff + tf + ft, a_min = 1, a_max = None)
pre_map = ft/np.clip(tf + ft, a_min = 1, a_max = None)

for i in range(len(rel_map)):
    rel_map[i, i] = 0
    pre_map[i, i] = 0

rel_map = rel_map*rel_filt
pre_map = pre_map*pre_filt


data = list()
for user in user_dict:
    seq = user_dict[user]
    while len(seq) >= 10:
        data.append(seq[:100])
        seq = seq[100:]

n_probs = len(prob_map)
n_knows = len(know_map)
dataset = dict()
dataset['n_probs'] = n_probs
dataset['n_knows'] = n_knows
dataset['n_feats'] = 5
dataset['data'] = data
dataset['rel_map'] = rel_map
dataset['pre_map'] = pre_map

import pickle
with open('assist09.bin', 'wb') as f:
    pickle.dump(dataset, f)
