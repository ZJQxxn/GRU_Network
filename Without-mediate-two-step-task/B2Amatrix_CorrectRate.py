import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import scipy.io as sio
import json
import h5py
import seaborn as sbn
import matplotlib.pyplot as plt



with open('matrix_mapping.json') as json_file:
    matrix_mapping = json.load(json_file)

temp_mapping = {}
for each in matrix_mapping:
    temp_mapping[each] = list(np.array(matrix_mapping[each]).T)
matrix_mapping = temp_mapping

# # show trial
# sbn.set(font_scale=1.6)
# y_lables = ['show A1', 'show A2', 'see nothing', 'do nothing', 'choose A1',
#                 'choose A2', 'reward', 'no reward']
# sbn.heatmap(np.array(np.array(matrix_mapping['A2NR']).T), cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
# plt.show()
# print()

filename = 'MyCode-validation-two_step_without_intermediate.hdf5'
# filename = 'SeqCode-20200130_1042-without-mediate.hdf5'

data = {}
with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    for k in list(f.keys()):
        # Get the data
        data[k]= list(f[k])


# In[322]:


# only select matrix with 13 time points
good_index = np.array([list(range(*i[0])) for i in data['index'] if i[0][1] - i[0][0] == 13 and
                      any(np.array_equal(np.array(data['behavior'][i[0][0]:i[0][1]]), v) for v in matrix_mapping.values())])  #4928 trials
print(len(good_index))
# good_index = np.array([list(range(*i[0])) for i in data['index'] if i[0][1] - i[0][0] == 13]) # decard the mapping matrix

# reshape to (4928, 13, 10), 4928 is trial numbers, 13 is time stamps, 10 is # of variables
all_trials_raw = np.array(data['behavior'])[good_index.ravel(),:].reshape(-1, 13, 8)

# pinpoint start points of each session. (session: contiguous matrices selected from raw data)
starts = [idx + 1 for idx, item in enumerate(np.diff(good_index[:, 0])) if item != 13]

# re-construct matrices
all_trials = [] # smaller than all_trials_raw, different is the len(starts)
for t in range(1, all_trials_raw.shape[0]):
    if t not in starts:
        all_trials.append(np.concatenate((all_trials_raw[t, :7, :], all_trials_raw[t - 1, 7:, :]), 0))
all_trials = np.array(all_trials)


# In[323]:


# categories each matrix and pick good matrix indexes
all_cnts, categories = [], []
for cnt in range(all_trials.shape[0]):
    v = [i for i in matrix_mapping.keys() if np.sum(all_trials[cnt, :] == matrix_mapping[i]) == 104] # 104 = 13 * 8
    if v != []:
        all_cnts.append(cnt)
        categories.append(v[0])
categories = np.array(categories) # smaller than all_trials, different is the len(empty v)


# In[324]:


block_size = 70
# block_size = int(139 // 2)
which_prob_big = np.tile(np.repeat([1, 0], block_size), 5000//(2 * block_size) + 1)[:5000]
repeat_cnt = np.array(data['index']).squeeze()[:,1] - np.array(data['index']).squeeze()[:,0]
switch = []
for i in range(5000):
    switch.extend([which_prob_big[i]] * repeat_cnt[i])
big_prob_raw = np.array(switch)[good_index]
connected_index = [idx for idx in range(1, np.array(switch)[good_index].shape[0]) if idx not in starts]
big_prob = big_prob_raw[connected_index][all_cnts,:][:,0]

del data

a, r = pd.Series(categories).str[:2].values, pd.Series(categories).str[2:].values
start_end = [0] + list((pd.Series(big_prob) - pd.Series(big_prob).shift(-1)).where(lambda x: x != 0).dropna().index + 1) + [len(categories)]
choice_matrix = pd.DataFrame([a[start_end[i]:start_end[i + 1]] for i in range(len(start_end) - 1)])

# choice_matrix = np.array(choice_matrix.values).reshape((-1, 100))
# choice_matrix = np.array(choice_matrix.values).reshape((-1, 140)) #TODO: rearrange block size

# Handle with previous
# previous = None
# for i in range(choice_matrix.shape[0]):
#     for j in range(choice_matrix.shape[1]):
#         if choice_matrix[i,j] != None:
#             previous = choice_matrix[i,j]
#         else:
#             choice_matrix[i, j] = previous


# first_block = np.equal(choice_matrix[list(range(0, 139, 2)),:], np.tile(['A1'], (block_size+1,36))).astype(int)
# second_block = np.equal(choice_matrix[list(range(1, 139, 2)),:],  np.tile(['A2'], (block_size, 36))).astype(int)
# first_block = np.mean(np.array(first_block).astype(int), axis=0)
# second_block = np.mean(np.array(second_block).astype(int), axis=0)

print()
choice_matrix = np.array(choice_matrix.values).reshape((-1, choice_matrix.shape[1]*2))
block_size = choice_matrix.shape[1] #TODO: rearrange block size; this is in fact 2 * block size

check = np.hstack((np.tile(['A1'], block_size // 2), np.tile(['A2'], block_size - block_size // 2)))
check = np.tile(check, (choice_matrix.shape[0], 1))
match = np.equal(choice_matrix, check).astype(int)
prob = np.mean(match, axis = 0)
# plt.plot(np.arange(0, 140, 1), prob)
plt.plot(np.arange(0, block_size, 1), prob,lw=2)
plt.plot([block_size // 2, block_size // 2], [0, 1], 'k--', lw = 1)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize = 30)
plt.xticks(fontsize = 30)
plt.show()

print(prob)
# sio.savemat('CorrectRate_'+filename.split('/')[-1].split('.')[0]+'.mat',
#             {'cr_matrix':np.array(pd.DataFrame(match_matrix).T)})

