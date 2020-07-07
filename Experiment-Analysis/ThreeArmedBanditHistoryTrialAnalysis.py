'''
Description: 
    Analyze the influence of history trials.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    May. 7 2020
'''

from scipy.io import loadmat
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression


class HistoryTrialAnalysis:

    def __init__(self, logFileName1, validationFileName1, logFileName2, validationFileName2, block_size = 150):
        self.block_size = block_size
        self.logFile1 = h5py.File(logFileName1, 'r')
        self.validationFileName1 = validationFileName1
        self.logFile2 = h5py.File(logFileName2, 'r')
        self.validationFileName2 = validationFileName2

    def influenceAnalysis(self):
        influence_matrix1, indication_matrix1, influence_matrix2, indication_matrix2  = self._constructInfluenceMatrix() # with shape of (3, number of trials, 6, 6)\
        # ============= PLOT WEIGHT MATRIX FOR RAN1===================
        coeff = np.zeros((6, 6))
        for stimulus in [0, 1, 2]:
            logstic_model = LogisticRegression().fit(influence_matrix1[stimulus], indication_matrix1[stimulus])
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
        # ============= PLOT WEIGHT MATRIX FOR RAN2===================
        coeff = np.zeros((6, 6))
        for stimulus in [0, 1, 2]:
            logstic_model = LogisticRegression().fit(influence_matrix2[stimulus], indication_matrix2[stimulus])
            coeff += np.abs(logstic_model.coef_.squeeze().reshape((6, 6)))  # TODO: need abs?
        coeff /= 3
        coeff = coeff / np.linalg.norm(coeff)
        # coeff = (coeff - np.mean(coeff)) / np.std(coeff)
        sbn.set(font_scale=1.6)
        labels = ['n-1', 'n-2', 'n-3', 'n-4', 'n-5']
        plt.figure(figsize=(14, 10))
        sbn.heatmap(coeff[1:, 1:][::-1, ::-1], cmap="binary", linewidths=0.5, xticklabels=labels,
                    yticklabels=labels)  # TODO: reverse the matrix because the weight begins from n-5 to n-1
        plt.ylabel('CHOICE', fontsize=20)
        plt.xlabel('REWARD', fontsize=20)
        plt.show()

        # ============= PLOT CHOICE REWARD INFLUENCE =========
        split_num = influence_matrix1.shape[1] // 1000 if influence_matrix1.shape[1] % 1000 == 0 \
            else influence_matrix1.shape[1] // 1000 + 1
        # for RAN1
        coeff_set1 = np.zeros((split_num, 36))
        for i in range(split_num):
            sub_influence_matrix = influence_matrix1[:, i * 1000:(i+1) * 1000, :]
            sub_indication_matrix = indication_matrix1[:, i * 1000:(i+1) * 1000]
            for stimulus in [0, 1, 2]:
                logstic_model = LogisticRegression(fit_intercept=False, solver="liblinear")\
                    .fit(sub_influence_matrix[stimulus], sub_indication_matrix[stimulus])
                coeff_set1[i, :] += logstic_model.coef_.squeeze()
            coeff_set1[i, :] /= 3
            coeff_set1[i, :] = coeff_set1[i, :] / np.linalg.norm(coeff_set1[i,:])
        # Compute mean and SEM (standard deviation / sqrt of sample size) values for every coefficient weight
        coeff_mean1 = np.mean(coeff_set1, axis = 0).reshape((6, 6))[1:, 1:] # Discard the relationship of n-6 trial
        coeff_mean1 = coeff_mean1[::-1, ::-1]
        coeff_SEM1 = sem(coeff_set1, axis = 0).reshape((6, 6))[1:, 1:] # Discard the relationship of n-6 trial
        coeff_SEM1 = coeff_SEM1[::-1, ::-1]
        # for RAN2
        coeff_set2 = np.zeros((split_num, 36))
        for i in range(split_num):
            sub_influence_matrix = influence_matrix2[:, i * 1000:(i + 1) * 1000, :]
            sub_indication_matrix = indication_matrix2[:, i * 1000:(i + 1) * 1000]
            for stimulus in [0, 1, 2]:
                logstic_model = LogisticRegression(fit_intercept=False, solver="liblinear") \
                    .fit(sub_influence_matrix[stimulus], sub_indication_matrix[stimulus])
                coeff_set2[i, :] += logstic_model.coef_.squeeze()
            coeff_set2[i, :] /= 3
            coeff_set2[i, :] = coeff_set2[i, :] / np.linalg.norm(coeff_set2[i, :])
        # Compute mean and SEM (standard deviation / sqrt of sample size) values for every coefficient weight
        coeff_mean2 = np.mean(coeff_set2, axis=0).reshape((6, 6))[1:, 1:]  # Discard the relationship of n-6 trial
        coeff_mean2 = coeff_mean2[::-1, ::-1]
        coeff_SEM2 = sem(coeff_set2, axis=0).reshape((6, 6))[1:, 1:]  # Discard the relationship of n-6 trial
        coeff_SEM2 = coeff_SEM2[::-1, ::-1]
        # choice - reward weight
        plt.errorbar(np.arange(0, 5, 1), coeff_mean1.diagonal(), yerr=coeff_SEM1.diagonal(),
                     lw = 2, ms = 12, fmt = '-ok', capsize = 6, label = 'reward affects the next one (RAN1)')
        plt.errorbar(np.arange(0, 5, 1), coeff_mean2.diagonal(), yerr=coeff_SEM2.diagonal(),
                     lw=2, ms=12, fmt='-sb', capsize=6, label = 'reward affects the next one (RAN2)')
        plt.xlabel('Trial in the past', fontsize = 20)
        plt.ylabel('Choice x Reward Weight', fontsize = 20)
        plt.xticks(np.arange(0, 5, 1), ['n-1', 'n-2', 'n-3', 'n-4', 'n-5'], fontsize = 20)
        # plt.ylim((-0.1,1))
        plt.yticks(fontsize = 20)
        plt.legend(loc = "upper right", fontsize = 25)
        plt.show()
        # past choice - immediate previous reward weight
        plt.errorbar(np.arange(0, 1, 1), coeff_mean1[0, 0], yerr=coeff_SEM1[0, 0],
                     lw=2, ms=12, fmt='-ok', capsize=6, )
        plt.errorbar(np.arange(1, 5, 1), coeff_mean1[1:, 0], yerr=coeff_SEM1[1:, 0],
                     lw=2, ms=12, fmt='-ok', capsize=6, label = 'reward affects the next one (RAN1)')
        plt.errorbar(np.arange(0, 1, 1), coeff_mean2[0, 0], yerr=coeff_SEM2[0, 0],
                     lw=2, ms=12, fmt='-sb', capsize=6)
        plt.errorbar(np.arange(1, 5, 1), coeff_mean2[1:, 0], yerr=coeff_SEM2[1:, 0],
                     lw=2, ms=12, fmt='-sb', capsize=6, label = 'reward affects the next two (RAN2)')
        plt.xlabel('Trial in the past', fontsize=20)
        plt.ylabel('Past Choices x Immediate Previous Reward Weight', fontsize=20)
        plt.xticks(np.arange(0, 5, 1), ['n-1', 'n-2', 'n-3', 'n-4', 'n-5'], fontsize=20)
        # plt.ylim((-0.1,1))
        plt.yticks(fontsize=20)
        plt.legend(loc = "upper right", fontsize = 25)
        plt.show()
        # immediate previous choice - past reward weight
        plt.errorbar(np.arange(0, 1, 1), coeff_mean1[0, 0], yerr=coeff_SEM1[0, 0],
                     lw=2, ms=12, fmt='-ok', capsize=6)
        plt.errorbar(np.arange(1, 5, 1), coeff_mean1[0,1:], yerr=coeff_SEM1[0,1:],
                     lw=2, ms=12, fmt='-ok', capsize=6, label = 'reward affects the next one (RAN1)')
        plt.errorbar(np.arange(0, 1, 1), coeff_mean2[0, 0], yerr=coeff_SEM2[0, 0],
                     lw=2, ms=12, fmt='-sb', capsize=6)
        plt.errorbar(np.arange(1, 5, 1), coeff_mean2[0, 1:], yerr=coeff_SEM2[0, 1:],
                     lw=2, ms=12, fmt='-sb', capsize=6, label = 'reward affects the next two (RAN2)')
        plt.xlabel('Trial in the past', fontsize=20)
        plt.ylabel('Immediate previous Choice x Past Rewards Weight', fontsize=20)
        plt.xticks(np.arange(0, 5, 1), ['n-1', 'n-2', 'n-3', 'n-4', 'n-5'], fontsize=20)
        # plt.ylim((-0.1,1))
        plt.yticks(fontsize=20)
        plt.legend(loc = "upper right", fontsize = 25)
        plt.show()

    def _constructInfluenceMatrix(self):
        choice1, reward1, choice2, reward2 = self._getChoiceAndReward()
        clip = choice1.shape[0]
        # For the RAN1
        choice1, reward1 = choice1[:clip,:], reward1[:clip, :]
        influence_matrix1 = np.zeros((3, clip-6, 36))
        indication_matrix1 = np.zeros((3, clip-6))
        for stimulus in [0, 1, 2]: # The choice is 1/2/3 while the index is 0/1/2
            for trial_index in range(6, clip):
                indication_matrix1[stimulus, trial_index-6] = int((stimulus + 1) == choice1[trial_index]) # indication of this trial
                for i in range(6): # n-1 to n-6 trials choices
                    for j in range(6): # n-1 to n-6 trials rewards
                        history_choice = choice1[trial_index-6+i, :]
                        history_reward = reward1[trial_index-6+j, :]
                        if history_choice == (stimulus+1) and history_reward == 1:
                            influence = 1
                        elif history_choice != (stimulus+1) and history_reward == 1:
                            influence = -1
                        else:
                            influence = 0
                        influence_matrix1[stimulus, trial_index-6, i*6+j] = influence

        # For the RAN1
        choice2, reward2 = choice2[:clip, :], reward2[:clip, :]
        influence_matrix2 = np.zeros((3, clip - 6, 36))
        indication_matrix2 = np.zeros((3, clip - 6))
        for stimulus in [0, 1, 2]:  # The choice is 1/2/3 while the index is 0/1/2
            for trial_index in range(6, clip):
                indication_matrix2[stimulus, trial_index - 6] = int(
                    (stimulus + 1) == choice2[trial_index])  # indication of this trial
                for i in range(6):  # n-1 to n-6 trials choices
                    for j in range(6):  # n-1 to n-6 trials rewards
                        history_choice = choice2[trial_index - 6 + i, :]
                        history_reward = reward2[trial_index - 6 + j, :]
                        if history_choice == (stimulus + 1) and history_reward == 1:
                            influence = 1
                        elif history_choice != (stimulus + 1) and history_reward == 1:
                            influence = -1
                        else:
                            influence = 0
                        influence_matrix2[stimulus, trial_index - 6, i * 6 + j] = influence
        return influence_matrix1, indication_matrix1, influence_matrix2, indication_matrix2

    def _getChoiceAndReward(self):
        choice1 = self.logFile1['choice']
        reward1= self.logFile1['reward']
        choice2 = self.logFile2['choice']
        reward2 = self.logFile2['reward']
        return choice1, reward1, choice2, reward2


if __name__ == '__main__':
    analyzer = HistoryTrialAnalysis('../RewardAffectData-OldTraining-OldNetwork-Three-Armed-Bandit/RewardAffectData-OldTraining-OldNetwork-ThreeArmed-sudden-reverse-model1-validation-1e6.hdf5',
                            '../RewardAffectData-OldTraining-OldNetwork-Three-Armed-Bandit/data/RewardAffect_ThreeArmed_TestingSet-2020_05_01-blk70-reverseblk0-noise-1.mat',
                            '../RewardAffectData-NewTraining-OldNetwork-Three-Armed-Bandit/RewardAffectData-NewTraining-OldNetwork-Three-Armed-sudden-reverse-model1-validation-1e6.hdf5',
                            '../RewardAffectData-NewTraining-OldNetwork-Three-Armed-Bandit/data/RewardAffect_ThreeArmed_TestingSet-2020_05_01-blk70-reverseblk0-noise-1.mat',
                            block_size = 70)

    analyzer.influenceAnalysis()
