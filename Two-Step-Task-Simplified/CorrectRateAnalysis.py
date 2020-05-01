'''
Description: 
    Compute the correct rate of the vaslidation result.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    10 Feb. 2020
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt


blk_size = 70

# choices of each validation trial
filename = 'MyCode-without-intermediate-less-time.hdf5' # withoutB, trans_prob = 1.0, less time
# filename = 'SeqCode-20200210_1310-revrl_lr.hdf5' # less time step; does not show choices

with h5py.File(filename, 'r') as f:
    choices = np.array(f['choice'].value, dtype = np.float32)
choices[np.where(choices > 2)] = np.nan

# the number of trials
numTrials = len(choices)
numBlks = numTrials // blk_size if (numTrials // blk_size) % 2 == 0 else (numTrials // blk_size) - 1 # number of small blocks
numTrials = numBlks * blk_size
choices = choices[:numTrials]

# A1 is reward in the first block; A2 is reward in the second block
block_indication = np.hstack(
    (np.zeros(blk_size,), np.ones(blk_size,))
)
block_indication = np.tile(block_indication, numBlks // 2)

# find out where the block reverse
block_sep = np.where(np.diff(block_indication) !=0 )[0]
numWholeBlocks = len(block_sep)

# correct rate
correct_rate = []
for whole_blk_index in range(numWholeBlocks - 1):
    reverse_point = block_sep[whole_blk_index]
    # next_reverse_points = block_sep[whole_blk_index+1]
    cr_range = np.arange(reverse_point - 10, reverse_point + 30)
    cr = [
        1 if choices[trial] == block_indication[trial] + 1 # choice is 1/2, indication is 0/1
        else 0
        for trial in cr_range
    ]
    correct_rate.append(cr)
correct_rate = np.vstack(correct_rate)
correct_rate = np.nanmean(correct_rate, axis = 0)

# show correct rate around the reverse point
plt.title('Correct Rate vs. Trial', fontsize = 30)
# plt.xticks(np.arange(-10,31), fontsize = 10)
plt.plot(np.arange(len(correct_rate)), correct_rate)
plt.xlabel('lag (trials)', fontsize = 30)
plt.ylabel('correct rate', fontsize = 30)
plt.yticks(np.linspace(0.0, 1.0, num=11, endpoint = True), fontsize = 30)
plt.xticks(np.arange(0, 41, 5), np.arange(0, 41, 5)-10, fontsize = 20)
plt.show()
