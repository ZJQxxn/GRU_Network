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
    #TODO: revise for three-reverse task

    def __init__(self, logFileName, validationFileName, block_size = 150):
        self.block_size = block_size
        self.logFile = h5py.File(logFileName, 'r')
        self.validationFileName = validationFileName

    def behaviorAnalysis(self):
        self.block_reward_prob = self._getBlockRewrdProbability()  # reward probability for all three stimulus in the 2 block
        self.objective_highest = self._getObjectiveHighest(self.block_reward_prob)  # the objective highest value stimulus (H_sch)
        self.expeienced_reward_prob = self._getExperiencedRewardProb() # the experienced reward probability
        self.random_reward = self._getRandomRewardProb(self.block_reward_prob, self.expeienced_reward_prob.shape)
        # Compute the mean and SEM over blocks (300 trial is one block)
        self.mean_experienced_reward_prob = np.nanmean(self.expeienced_reward_prob, axis = 1) # average value
        self.SEM_experienced_reward_prob = sem(self.expeienced_reward_prob, axis = 1) # SEM
        # # TODO: for moving window
        # self.mean_experienced_reward_prob = self.expeienced_reward_prob[:,0]  # average value
        # self.SEM_experienced_reward_prob = self.expeienced_reward_prob[:,1] # SEM
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
        # plt.fill_between(np.arange(0, self.block_size * 2),
        #                  self.mean_experienced_reward_prob - self.SEM_experienced_reward_prob,
        #                  self.mean_experienced_reward_prob + self.SEM_experienced_reward_prob,
        #                  color = "#dcb2ed",
        #                  alpha = 0.8,
        #                  linewidth = 4)
        plt.plot(np.mean(self.random_reward, axis = 1), 'b--', alpha = 0.5,
                 label = "Random Reward Prob.", lw = 2)
        plt.ylim((0.0, 0.85))
        plt.yticks(fontsize = 20)
        plt.ylabel("Reward Probability", fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.xlabel("Trial", fontsize = 20)
        plt.legend(loc = "best", fontsize = 20)
        plt.show()

    def influenceAnalysis(self):
        influence_matrix, indication_matrix = self._constructInfluenceMatrix() # with shape of (3, number of trials, 6, 6)\
        # ============= PLOT WEIGHT MATRIX ===================
        coeff = np.zeros((6, 6))
        for stimulus in [0, 1, 2]:
            logstic_model = LogisticRegression().fit(influence_matrix[stimulus], indication_matrix[stimulus])
            coeff += np.abs(logstic_model.coef_.squeeze().reshape((6,6))) # TODO: need abs?
        coeff /= 3
        coeff = coeff / np.linalg.norm(coeff)
        # coeff = (coeff - np.mean(coeff)) / np.std(coeff)
        sbn.set(font_scale=1.6)
        labels = ['n-1', 'n-2', 'n-3', 'n-4', 'n-5']
        plt.figure(figsize=(14,10))
        sbn.heatmap(coeff[1:,1:][::-1, ::-1], cmap="binary", linewidths=0.5, xticklabels=labels, yticklabels=labels) #TODO: reverse the matrix because the weight begins from n-5 to n-1
        plt.ylabel('CHOICE', fontsize=20)
        plt.xlabel('REWARD', fontsize=20)
        plt.show()
        # ============= PLOT CHOICE REWARD INFLUENCE =========
        split_num = influence_matrix.shape[1] // 1000 if influence_matrix.shape[1] % 1000 == 0 \
            else influence_matrix.shape[1] // 1000 + 1
        coeff_set = np.zeros((split_num, 36))
        for i in range(split_num):
            sub_influence_matrix = influence_matrix[:, i * 1000:(i+1) * 1000, :]
            sub_indication_matrix = indication_matrix[:, i * 1000:(i+1) * 1000]
            for stimulus in [0, 1, 2]:
                logstic_model = LogisticRegression(fit_intercept=False, solver="liblinear")\
                    .fit(sub_influence_matrix[stimulus], sub_indication_matrix[stimulus])
                coeff_set[i, :] += logstic_model.coef_.squeeze()
            coeff_set[i, :] /= 3
            coeff_set[i, :] = coeff_set[i, :] / np.linalg.norm(coeff_set[i,:])
        # Compute mean and SEM (standard deviation / sqrt of sample size) values for every coefficient weight
        coeff_mean = np.mean(coeff_set, axis = 0).reshape((6, 6))[1:, 1:] # Discard the relationship of n-6 trial
        coeff_mean = coeff_mean[::-1, ::-1]
        coeff_SEM = sem(coeff_set, axis = 0).reshape((6, 6))[1:, 1:] # Discard the relationship of n-6 trial
        coeff_SEM = coeff_SEM[::-1, ::-1]
        # choice - reward weight
        plt.errorbar(np.arange(0, 5, 1), coeff_mean.diagonal(), yerr=coeff_SEM.diagonal(),
                     lw = 2, ms = 12, fmt = '-ok', capsize = 6)
        plt.xlabel('Trial in the past', fontsize = 20)
        plt.ylabel('Choice x Reward Weight', fontsize = 20)
        plt.xticks(np.arange(0, 5, 1), ['n-1', 'n-2', 'n-3', 'n-4', 'n-5'], fontsize = 20)
        # plt.ylim((-0.1,1))
        plt.yticks(fontsize = 20)
        plt.show()
        # past choice - immediate previous reward weight
        plt.errorbar(np.arange(0, 1, 1), coeff_mean[0, 0], yerr=coeff_SEM[0, 0],
                     lw=2, ms=12, fmt='-ok', capsize=6)
        plt.errorbar(np.arange(1, 5, 1), coeff_mean[1:, 0], yerr=coeff_SEM[1:, 0],
                     lw=2, ms=12, fmt='-ok', capsize=6)
        plt.xlabel('Trial in the past', fontsize=20)
        plt.ylabel('Past Choices x Immediate Previous Reward Weight', fontsize=20)
        plt.xticks(np.arange(0, 5, 1), ['n-1', 'n-2', 'n-3', 'n-4', 'n-5'], fontsize=20)
        # plt.ylim((-0.1,1))
        plt.yticks(fontsize=20)
        plt.show()
        # immediate previous choice - past reward weight
        plt.errorbar(np.arange(0, 1, 1), coeff_mean[0, 0], yerr=coeff_SEM[0, 0],
                     lw=2, ms=12, fmt='-ok', capsize=6)
        plt.errorbar(np.arange(1, 5, 1), coeff_mean[0,1:], yerr=coeff_SEM[0,1:],
                     lw=2, ms=12, fmt='-ok', capsize=6)
        plt.xlabel('Trial in the past', fontsize=20)
        plt.ylabel('Immediate previous Choice x Past Rewards Weight', fontsize=20)
        plt.xticks(np.arange(0, 5, 1), ['n-1', 'n-2', 'n-3', 'n-4', 'n-5'], fontsize=20)
        # plt.ylim((-0.1,1))
        plt.yticks(fontsize=20)
        plt.show()

    def _constructInfluenceMatrix(self):
        choice, reward = self._getChoiceAndReward()
        clip = choice.shape[0]
        # clip = 140
        choice, reward = choice[:clip,:], reward[:clip, :]
        influence_matrix = np.zeros((3, clip-6, 36))
        indication_matrix = np.zeros((3, clip-6))
        for stimulus in [0, 1, 2]: # The choice is 1/2/3 while the index is 0/1/2
            for trial_index in range(6, clip):
                indication_matrix[stimulus, trial_index-6] = int((stimulus + 1) == choice[trial_index]) # indication of this trial
                for i in range(6): # n-1 to n-6 trials choices
                    for j in range(6): # n-1 to n-6 trials rewards
                        history_choice = choice[trial_index-6+i, :]
                        history_reward = reward[trial_index-6+j, :]
                        if history_choice == (stimulus+1) and history_reward == 1:
                            influence = 1
                        elif history_choice != (stimulus+1) and history_reward == 1:
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
        mat = loadmat(self.validationFileName)
        reward_prob = mat['data_ST_Brief']['reward_prob_1'][0, 0][:,:(2*self.block_size)]
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
        # extract the stimulus choices of all the trials
        trial_choice = self.logFile['choice']
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
    analyzer = TaskAnalyzer('ThreeReverse-severe_reverse_three_block-ThreeArmed-validation-15e6.hdf5',
                            './data/ThreeReverse_ThreeArmed_TestingSet-2020_07_24-severe_three_block-1.mat',
                            block_size = 75)

    analyzer.behaviorAnalysis()
    # analyzer.influenceAnalysis()
