'''
Description:
    Compare the choice distribution with the reward probability distribution.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date: 
    Apr. 6 2020
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat


blk_size = 70
whole_blk_size = 2 * blk_size
# Choices of each validation trial
filename =  'SimplifyTwoArmed-validation-75e5.hdf5'
with h5py.File(filename, 'r') as f:
    choices = np.array(f['choice'].value, dtype = np.float32)
choices[np.where(choices > 2)] = np.nan
numTrials = len(choices)
numWholeBlks = numTrials // whole_blk_size if (numTrials // whole_blk_size) % 2 == 0 \
    else (numTrials // whole_blk_size) - 1  # number of small blocks
numTrials = numWholeBlks * whole_blk_size
choices = choices[:numTrials]



# Reward probability
noise_filename = "data/SimplifyTwoArmed_TestingSet-2020_02_15-1.mat"
reward_prob = loadmat(noise_filename)['data_ST_Brief']['reward_prob_1'][0][0]
exp_reward = np.array(
    [reward_prob[int(each[0]) - 1 if not np.isnan(each[0]) else np.random.choice([0, 1], 1), index]
     for index, each in enumerate(choices)]
)
noise_exp_reward = exp_reward.reshape((numWholeBlks, -1))


# the choice distribution and reward prob distribution for "without noise"
plt.title("Choice Distribution vs. Reward Probability [sudden & no noise]", fontsize = 20)
plt.plot(reward_prob[0,:whole_blk_size], '-or', label = "A reward probability", lw=2, ms=12)
plt.plot(np.sum(choices.reshape((numWholeBlks, -1)) == 1, axis = 0) / numWholeBlks, '-ob', label = "A choice proportion",lw=2, ms=12)
plt.plot([70,70],[0,1],  '--k')
plt.xticks(fontsize = 20)
plt.xlabel("Trial", fontsize = 20)
plt.yticks(fontsize = 20)
plt.ylim((0, 1))
plt.ylabel("Reward Prob. / Choice Proportion", fontsize = 20)
plt.legend(loc = "best", fontsize = 20) # located at the upper center
plt.show()


