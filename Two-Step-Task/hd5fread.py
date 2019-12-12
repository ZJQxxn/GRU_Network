import h5py
import json
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


with open('matrix_mapping.json') as json_file:
    matrix_mapping = json.load(json_file)


with h5py.File('20191209_1134-smp_ts.hdf5', 'r') as file:
    key = file.keys()
    prediction = file['prediction'].value
    choice = file['choice'].value
    reward = file['reward'].value
    index_list = file['index'].value
    behavior = file['behavior'].value


good_index = np.array([trial_index for trial_index, index in enumerate(index_list) if index[0][1] - index[0][0] == 13 and
                       any(np.array_equal(np.array(behavior[index[0][0]:index[0][1]]), v) for v in matrix_mapping.values())])
choice = choice[good_index, :]
# reward = reward[good_index, :]


index = 0
good_trial_num = len(good_index)
large_block_size = 140
blk_num = good_trial_num // large_block_size
count = np.zeros((blk_num, large_block_size))
previous = 0
try:
    for step, trial in enumerate(choice):
        if step >= (blk_num * large_block_size):
            break
        blk_index = step // large_block_size
        if trial == 1:
            count[blk_index, step % large_block_size] = 1
            previous = 1
        elif trial == 2:
            count[blk_index, step % large_block_size] = 2
            previous = 2
        else:
            count[blk_index, step % large_block_size] = previous
except:
    print()
count = count.astype(int)

# count = count.flatten()[:-78].reshape((-1, 134))
# count = count.flatten()[135:-65].reshape((-1, 100))[:10]



I = np.ones((count.shape[0], count.shape[1] // 2))
c = np.hstack((1 * I, 2 * I))
match = np.array(count == c).astype(int)
prob = np.mean(match, axis=0)
plt.plot(np.arange(0, count.shape[1], 1), prob)
plt.yticks(np.arange(0, 1, 0.1))
plt.xlabel('trial number')
plt.ylabel('correct rate')
plt.show()
print()