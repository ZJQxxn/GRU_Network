'''
TaskAnalysis.py: Analysis of the three-armed bandit task with stacked GRU network.
'''

from scipy.io import loadmat
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression


class TaskAnalyzer:

    def __init__(self, logFileName1, logFileName2, logFileName3, validationFileName, block_size = 150):
        self.block_size = block_size
        self.logFile1 = h5py.File(logFileName1, 'r')
        self.logFile2 = h5py.File(logFileName2, 'r')
        self.logFile3 = h5py.File(logFileName3, 'r')
        self.validationFileName = validationFileName

    def behaviorAnalysis(self):
        self.block_reward_prob = self._getBlockRewrdProbability()  # reward probability for all three stimulus in the 2 block
        self.objective_highest = self._getObjectiveHighest(self.block_reward_prob)  # the objective highest value stimulus (H_sch)
        self.expeienced_reward_prob1_mean = np.nanmean(self._getExperiencedRewardProb(1), axis = 1) # the experienced reward probability for model 1
        self.expeienced_reward_prob2_mean = np.nanmean(self._getExperiencedRewardProb(2), axis = 1)  # the experienced reward probability for model 2
        self.expeienced_reward_prob3_mean = np.nanmean(self._getExperiencedRewardProb(3), axis = 1)  # the experienced reward probability for model 3
        self.random_reward = self._getRandomRewardProb(self.block_reward_prob, self._getExperiencedRewardProb().shape)
        # Compute the mean and SEM over blocks (300 trial is one block)
        self.SEM_experienced_reward_prob = sem(
            np.vstack((self.expeienced_reward_prob1_mean, self.expeienced_reward_prob2_mean, self.expeienced_reward_prob3_mean)),
            axis = 0) # SEM
        self.mean_experienced_reward_prob = np.nanmean(
            np.vstack((self.expeienced_reward_prob1_mean, self.expeienced_reward_prob2_mean,self.expeienced_reward_prob3_mean)),
            axis=0)  # the experienced reward probability for model 1
        # ================== PLOT PROBABILITY ==================
        # Show block reward probability
        plt.plot(np.arange(0, self.block_size * 2), self.block_reward_prob[0, :], 'o-r', label='stimulus A')
        plt.plot(np.arange(0, self.block_size * 2), self.block_reward_prob[1, :], 'o-b', label='stimulus B')
        plt.plot(np.arange(0, self.block_size * 2), self.block_reward_prob[2, :], 'o-g', label='stimulus C')
        plt.legend(fontsize=20, loc = "best")
        plt.show()
        # Show H_sch
        for i in range(2 * self.block_size):
            temp = self.objective_highest[i, :]
            if temp[0] == 0:
                color = 'red'
            elif temp[0] == 1:
                color = 'blue'
            elif temp[0] == 2:
                color = 'green'
            else:
                color = 'cyan'
            plt.scatter(i, temp[1], color=color)
        plt.title("Experienced Reward Prob. vs. Random Reward Prob.", fontsize = 20)
        plt.plot(np.arange(0, self.block_size * 2), self.objective_highest[:, 1], 'k-', ms = 8)
        # Plot experienced reward probability
        plt.plot(np.arange(0, self.block_size * 2), self.mean_experienced_reward_prob, 's-m',
                 label = "Experienced Reward Prob.", ms = 8, lw = 2)
        plt.fill_between(np.arange(0, self.block_size * 2),
                         self.mean_experienced_reward_prob - self.SEM_experienced_reward_prob,
                         self.mean_experienced_reward_prob + self.SEM_experienced_reward_prob,
                         color = "#dcb2ed",
                         alpha = 0.5,
                         linewidth = 4)
        plt.plot(np.mean(self.random_reward, axis = 1), 'b--', alpha = 0.5,
                 label = "Random Reward Prob.", lw = 2)
        plt.ylim((0.0, 0.85))
        plt.yticks(fontsize = 20)
        plt.ylabel("Reward Probability", fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.xlabel("Trial", fontsize = 20)
        plt.legend(loc = "best", fontsize = 20)
        plt.show()


    def _getChoiceAndReward(self, modelName = 1):
        if modelName == 1:
            choice = self.logFile1['choice']
            reward = self.logFile1['reward']
        elif modelName == 2:
            choice = self.logFile2['choice']
            reward = self.logFile2['reward']
        else:
            choice = self.logFile3['choice']
            reward = self.logFile3['reward']
        return choice, reward

    def _getBlockRewrdProbability(self):
        mat = loadmat(self.validationFileName)
        reward_prob = mat['data_ST_Brief']['reward_prob_1'][0, 0][:,:140]
        del mat
        return reward_prob

    def _getObjectiveHighest(self, reward_prob):
        objective_highest = []
        trial_num = reward_prob.shape[1]
        for i in range(trial_num):
            trial_reward = reward_prob[:, i]
            max_ind = np.argwhere(trial_reward == np.amax(trial_reward))
            max_ind = max_ind.flatten()
            if 1 == len(max_ind.shape):  # only one stimulus has the highest reward probability
                objective_highest.append([max_ind[0], trial_reward[max_ind[0]]])
            elif 2 == len(max_ind.shape):  # two stimulus has the highest reward probability
                # 3 for A/B, 4 for A/C, 5 for B/C
                highest_reward = trial_reward[0] if 0 in max_ind else trial_reward[1]
                objective_highest.append([np.sum(max_ind) + 2, trial_reward[0], highest_reward])
            else:  # all the stimulus has the same reward probability
                objective_highest.append([6, trial_reward[0]])
        return np.array(objective_highest)

    def _getExperiencedRewardProb(self, modelName = 1):
        # extract the stimulus choices of all the trials
        if modelName == 1:
            trial_choice = self.logFile1['choice']
        elif modelName == 2:
            trial_choice = self.logFile2['choice']
        else:
            trial_choice = self.logFile3['choice']
        trial_num = trial_choice.shape[0]
        block_num = trial_num // (self.block_size * 2)
        # Experienced reward probability os a (number of trials in one block, number of blocks) matrix
        choice_reward_prob = np.zeros((self.block_size * 2, block_num))
        for index, choice in enumerate(trial_choice):
            index_in_block = index % (2*self.block_size)
            block_index = index // (2*self.block_size)
            if block_index >= block_num:
                break
            if choice.item() > 3:
                choice_reward_prob[index_in_block, block_index] = np.nan
            else:
                choice_reward_prob[index_in_block, block_index] = self.block_reward_prob[choice.item()-1, index_in_block]
        return np.array(choice_reward_prob)

    def _getRandomRewardProb(self, block_reward, trial_shape):
        random_reward = np.array(
            [block_reward[np.random.choice([0, 1, 2], 1),  index % block_reward.shape[1]] for index in range(trial_shape[0] * trial_shape[1])])\
            .reshape(trial_shape)
        return random_reward


if __name__ == '__main__':
    analyzer = TaskAnalyzer(
        'RewardAffectData-NewTraining-OldNetwork-Three-Armed-slow-reverse-model1-validation-1e6.hdf5',
        'RewardAffectData-NewTraining-OldNetwork-Three-Armed-slow-reverse-model2-validation-1e6.hdf5',
        'RewardAffectData-NewTraining-OldNetwork-Three-Armed-slow-reverse-model3-validation-1e6.hdf5',
        './data/RewardAffect_ThreeArmed_TestingSet-2020_05_03-blk70-reverseblk5-noise-1.mat',
        block_size = 70
    )

    analyzer.behaviorAnalysis()
