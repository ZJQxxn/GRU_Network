'''
TaskAnalysis.py: Analysis of the three-armed bandit task.
'''

from scipy.io import loadmat
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression


class TaskAnalyzer:

    def __init__(self, logFileName):
        self.block_size = 150 #TODO: a new name for block size or let block size = 300
        self.logFile = h5py.File(logFileName, 'r')

    def behaviorAnalysis(self):
        self.block_reward_prob = self._getBlockRewrdProbability()  # reward probability for all three stimulus in the 2 block
        self.objective_highest = self._getObjectiveHighest(self.block_reward_prob)  # the objective highest value stimulus (H_sch)
        self.expeienced_reward_prob = self._getExperiencedRewardProb() # the experienced reward probability
        # Compute the mean and SEM over blocks (300 trial is one block)
        self.mean_experienced_reward_prob = np.mean(self.expeienced_reward_prob, axis = 1) # average value
        self.SEM_experienced_reward_prob = sem(self.expeienced_reward_prob, axis = 1) # SEM
        # # TODO: for moving window
        # self.mean_experienced_reward_prob = self.expeienced_reward_prob[:,0]  # average value
        # self.SEM_experienced_reward_prob = self.expeienced_reward_prob[:,1] # SEM

    def influenceAnalysis(self):
        influence_matrix, indication_matrix = self._constructInfluenceMatrix() # with shape of (3, number of trials, 6, 6)
        coeff = np.zeros((6, 6))
        for stimulus in [0, 1, 2]:
            logstic_model = LogisticRegression().fit(influence_matrix[stimulus], indication_matrix[stimulus])
            coeff += logstic_model.coef_.squeeze().reshape((6,6))
        coeff /= 3
        coeff = coeff / np.linalg.norm(coeff)
        # Print for test
        sbn.set(font_scale=1.6)
        labels = ['n-1', 'n-2', 'n-3', 'n-4', 'n-5']
        sbn.heatmap(coeff[:5,:5], cmap="Blues", linewidths=0.5, xticklabels=labels, yticklabels=labels)
        plt.show()

    def _constructInfluenceMatrix(self):
        choice, reward = self._getChoiceAndReward()
        #TODO: for test and for simplicity, choose first 100 trials
        clip = 3000
        choice, reward = choice[:clip,:], reward[:clip, :]
        trial_num = choice.shape[0]
        influence_matrix = np.zeros((3, clip-6, 36)) #TODO: notice the sahpe
        indication_matrix = np.zeros((3, clip-6))
        for stimulus in [0, 1, 2]:
            for trial_index in range(6, clip):
                indication_matrix[stimulus, trial_index-6] = int(stimulus==choice[trial_index]) # indication of this trial
                for i in range(6): # n-1 to n-6 trials choices
                    for j in range(6): # n-1 to n-6 trials rewards
                        history_choice = choice[i, :]
                        history_reward = reward[j, :]
                        if history_choice == stimulus and history_reward == 1:
                            influence = 1
                        elif history_choice != stimulus and history_reward == 1:
                            influence = -1
                        else:
                            influence = 0
                        influence_matrix[stimulus, trial_index-6, i*6+j] = influence
        return influence_matrix, indication_matrix


    def _getChoiceAndReward(self):
        choice = self.logFile['choice']
        reward = self.logFile['reward']
        return choice, reward

    def _getBlockRewrdProbability(self):
        mat = loadmat('./data/ThreeArmedBandit_TestingSet-reverse-2019_12_04-2.mat') #TODO: cahnge this
        reward_prob = mat['info']['reward_probability'][0, 0]
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

    def _getExperiencedRewardProb(self):
        # # TODO: poor result; change to use moving window (20 trials)
        # extract the stimulus choices of all the trials
        trial_choice = self.logFile['choice']
        choice_count = [0, 0, 0]
        for each in trial_choice[751:900]:
            choice_count[each[0]] += 1
        trial_num = trial_choice.shape[0]
        block_num = trial_num // (self.block_size * 2)
        # Experienced reward probability os a (number of trials in one block, number of blocks) matrix
        choice_reward_prob = np.zeros((self.block_size * 2, block_num))
        for index, choice in enumerate(trial_choice):
            index_in_block = index % (2*self.block_size)
            block_index = index // (2*self.block_size)
            choice_reward_prob[index_in_block, block_index] = self.block_reward_prob[choice[0], index_in_block]
        # TODO: moving window ;there are 17 blocks in total, take the 16-th block
        # trial_choice = self.logFile['choice']
        # sti_count = [0, 0, 0]
        # for each in trial_choice:
        #     sti_count[each[0]] += 1
        # trial_num = trial_choice.shape[0]
        # block_num = trial_num // (self.block_size * 2)
        # choice_reward_prob = []
        # block_start = (block_num-2) * 300
        # for within_index in range(300):
        #     cur_trial_index = block_start + within_index
        #     temp_choice = trial_choice[cur_trial_index-10:cur_trial_index+10,:].squeeze()
        #     temp_reward = np.array([self.block_reward_prob[temp_choice[i], within_index] for i in range(20)])
        #     choice_reward_prob.append([np.mean(temp_reward), sem(temp_reward)])
        # # #     #print()
        return np.array(choice_reward_prob)

    def plotFigures(self):
        # Show block reward probability
        plt.plot(np.arange(0, self.block_size * 2), self.block_reward_prob[0, :], 'o-r', label='stimulus A')
        plt.plot(np.arange(0, self.block_size * 2), self.block_reward_prob[1, :], 'o-b', label='stimulus B')
        plt.plot(np.arange(0, self.block_size * 2), self.block_reward_prob[2, :], 'o-g', label='stimulus C')
        plt.legend(fontsize=20)
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
            plt.scatter(i, temp[1], color=color)  # TODO: split into groups, then plot scatters with labels
        plt.plot(np.arange(0, self.block_size * 2), self.objective_highest[:, 1], 'k-')
        #plt.yticks(np.arange(0.0, 1.0, 0.1))
        #plt.show()
        # Plot experienced reward probability
        plt.plot(np.arange(0, self.block_size * 2), self.mean_experienced_reward_prob, '*-m')
        plt.yticks(np.arange(0.0, 1.0, 0.1))
        plt.show()




if __name__ == '__main__':
    analyzer = TaskAnalyzer('validate_record-three-armed-2019_12_04-5e4-without_init.hdf5')
    analyzer.behaviorAnalysis()
    analyzer.plotFigures()
    # analyzer.influenceAnalysis()
