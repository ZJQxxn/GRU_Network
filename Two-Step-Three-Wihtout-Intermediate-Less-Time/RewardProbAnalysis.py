from scipy.io import loadmat
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression

blk_size = 70

# choices of each validation trial
filename = 'MyCode-TwoStepThreeLess-15e6-validation.hdf5' # TODO: can't lean the reversal property amongthree stimulus
# filename = 'SeqCode-20200210_1310-revrl_lr.hdf5' # less time step; does not show choices

with h5py.File(filename, 'r') as f:
    choices = np.array(f['choice'].value, dtype = np.float32)
choices[np.where(choices > 3)] = np.nan

# the number of trials
numTrials = len(choices)
numBlks = numTrials // (2*blk_size) if (numTrials // (2*blk_size)) % 2 == 0 else (numTrials // (2*blk_size)) - 1 # number of small blocks
numTrials = numBlks * 2* blk_size
choices = choices[:numTrials]

# reward probability
first_block_reward_prob = [0.8, 0.5, 0.2]
second_block_reward_prob = [0.2, 0.5, 0.8]
one_whole_block_reward = np.hstack((np.tile(np.array(first_block_reward_prob).reshape((3,1)), blk_size),
                                    np.tile(np.array(second_block_reward_prob).reshape((3, 1)), blk_size))
                                   )
reward_prob = np.tile(one_whole_block_reward, numBlks).reshape(-1)

# objective reward probability
choice_reward_prob = np.zeros(choices.shape)
for index, choice in enumerate(choices):
    if np.nan == choice:
        choice_reward_prob[index] = np.nan
    else:
        choice_reward_prob[index] = reward_prob[index]
choice_reward_prob = choice_reward_prob.reshape((numBlks, blk_size*2))
mean_choice_reward_prob = np.nanmean(choice_reward_prob, axis = 0)

# show objective reward probability
plt.title('Objective Reward Probability vs. Trial', fontsize = 30)
# plt.xticks(np.arange(-10,31), fontsize = 10)
plt.plot(np.arange(len(mean_choice_reward_prob)), mean_choice_reward_prob)
plt.xlabel('trial', fontsize = 30)
plt.ylabel('reward prob', fontsize = 30)
plt.yticks(np.linspace(0.0, 1.0, num=11, endpoint = True), fontsize = 30)
plt.xticks(np.arange(0, 2*blk_size+1, 20), np.arange(0, 2*blk_size+1, 20), fontsize = 20)
plt.show()