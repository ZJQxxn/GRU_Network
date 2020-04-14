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
        self.block_size = 70 #TODO: a new name for block size or let block size = 300
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
        # Show block reward probability
        plt.plot(np.arange(0, self.block_size * 2), self.block_reward_prob[0, :], 'o-r', label='stimulus A1')
        plt.plot(np.arange(0, self.block_size * 2), self.block_reward_prob[1, :], 'o-b', label='stimulus A2')
        plt.legend(fontsize=20)
        plt.show()
        # Show H_sch
        for i in range(2 * self.block_size):
            temp = self.objective_highest[i, :]
            if temp[0] == 0: # choose A1
                color = 'red'
            elif temp[0] == 1: # choose A2
                color = 'blue'
            else:
                pass
            plt.scatter(i, temp[1], color=color)  # TODO: split into groups, then plot scatters with labels
        plt.plot(np.arange(0, self.block_size * 2), self.objective_highest[:, 1], 'k-')
        # plt.yticks(np.arange(0.0, 1.0, 0.1))
        # plt.show()
        # Plot experienced reward probability
        plt.plot(np.arange(0, self.block_size * 2), self.mean_experienced_reward_prob, '*-m')
        plt.yticks(np.arange(0.0, 1.0, 0.1))
        plt.show()

    def influenceAnalysis(self):
        influence_matrix, indication_matrix = self._constructInfluenceMatrix() # with shape of (2, number of trials, 6, 6)\
        # ============= PLOT WEIGHT MATRIX ===================
        coeff = np.zeros((6, 6))
        for index, stimulus in enumerate([1, 2]):
            logstic_model = LogisticRegression().fit(influence_matrix[index], indication_matrix[index])
            coeff += np.abs(logstic_model.coef_.squeeze().reshape((6,6))) # TODO: need abs?
        coeff /= 3
        coeff = coeff / np.linalg.norm(coeff)
        # coeff = (coeff - np.mean(coeff)) / np.std(coeff)
        sbn.set(font_scale=1.6)
        labels = ['n-1', 'n-2', 'n-3', 'n-4', 'n-5']
        sbn.heatmap(coeff[1:,1:][::-1, ::-1], cmap="binary", linewidths=0.5, xticklabels=labels, yticklabels=labels) #TODO: reverse the matrix because the weight begins from n-5 to n-1
        plt.ylabel('CHOICE', fontsize=20)
        plt.xlabel('REWARD', fontsize=20)
        plt.show()
        # ============= PLOT CHOICE REWARD INFLUENCE =========
        split_num = influence_matrix.shape[1] // 1000 if influence_matrix.shape[1] % 1000 == 0 \
            else influence_matrix.shape[1] // 1000 + 1
        coeff_set = np.zeros((split_num, 36))
        for i in range(split_num):
            sub_influence_matrix = influence_matrix[:,i*1000:(i+1)*1000,:]
            sub_indication_matrix = indication_matrix[:,i*1000:(i+1)*1000]
            for index, stimulus in enumerate([1, 2]):
                logstic_model = LogisticRegression().fit(sub_influence_matrix[index], sub_indication_matrix[index])
                coeff_set[i, :] += np.abs(logstic_model.coef_.squeeze())
            coeff_set[i, :] /= 3
            coeff_set[i, :] = coeff_set[i, :] / np.linalg.norm(coeff_set[i,:])
        # Compute mean and SEM (standard deviation / sqrt of sample size) values for every coefficient weight
        coeff_mean = np.mean(coeff_set, axis = 0).reshape((6, 6))[1:, 1:] # Discard the relationship of n-6 trial
        coeff_SEM = sem(coeff_set, axis = 0).reshape((6, 6))[1:, 1:] # Discard the relationship of n-6 trial
        # choice - reward weight
        plt.errorbar(np.arange(0, 5, 1), coeff_mean.diagonal()[::-1], yerr=coeff_SEM.diagonal()[::-1], #TODO: reverse the array
                     lw = 2, ms = 12, fmt = '-ok', capsize = 6)
        plt.xlabel('Trial in the past', fontsize = 20)
        plt.ylabel('Choice x Reward Weight', fontsize = 20)
        plt.xticks(np.arange(0, 5, 1), ['n-1', 'n-2', 'n-3', 'n-4', 'n-5'], fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.show()
        # past choice - immediate previous reward weight
        plt.errorbar(np.arange(0, 5, 1), coeff_mean[:,0][::-1], yerr=coeff_SEM[:,0][::-1],
                     lw=2, ms=12, fmt='-ok', capsize=6)
        plt.xlabel('Trial in the past', fontsize=20)
        plt.ylabel('Past Choices x Immediate Previous Reward Weight', fontsize=20)
        plt.xticks(np.arange(0, 5, 1), ['n-1', 'n-2', 'n-3', 'n-4', 'n-5'], fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()
        # immediate previous choice - past reward weight
        plt.errorbar(np.arange(0, 5, 1), coeff_mean[0,:][::-1], yerr=coeff_SEM[0,:][::-1],
                     lw=2, ms=12, fmt='-ok', capsize=6)
        plt.xlabel('Trial in the past', fontsize=20)
        plt.ylabel('Immediate previous Choice x Past Rewards Weight', fontsize=20)
        plt.xticks(np.arange(0, 5, 1), ['n-1', 'n-2', 'n-3', 'n-4', 'n-5'], fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

    def _constructInfluenceMatrix(self):
        choice, reward = self._getChoiceAndReward()
        #TODO: for test and for simplicity, choose first 100 trials
        clip = choice.shape[0]
        choice, reward = choice[:clip,:], reward[:clip, :]
        influence_matrix = np.zeros((2, clip-6, 36)) #TODO: notice the sahpe
        indication_matrix = np.zeros((2, clip-6))
        for index, stimulus in enumerate([1, 2]):
            for trial_index in range(6, clip):
                indication_matrix[index, trial_index-6] = int(stimulus == choice[trial_index]) # indication of this trial
                for i in range(6): # n-1 to n-6 trials choices
                    for j in range(6): # n-1 to n-6 trials rewards
                        history_choice = choice[trial_index-6+i, :]
                        history_reward = reward[trial_index-6+j, :]
                        if history_choice == stimulus and history_reward == 1:
                            influence = 1
                        elif history_choice != stimulus and history_reward == 1:
                            influence = -1
                        else:
                            influence = 0
                        influence_matrix[index, trial_index-6, i*6+j] = influence
        return influence_matrix, indication_matrix

    def _getChoiceAndReward(self):
        choice = np.array(self.logFile['choice'])

        # count  = 0
        # total_count = 0
        # for each in choice:
        #     if each.item() == 1 or each.item() == 2:
        #         total_count += 1
        #     if each.item() == 2:
        #         count += 1
        # ratio = count / total_count

        reward = np.array(self.logFile['reward'])
        trial_num = choice.shape[0]
        good_index = []
        for i in range(trial_num):
            if choice[i] in [1,2]:
                good_index.append(i)
        choice = choice[good_index, :]
        reward = reward[good_index, :]
        self.choice = choice
        self.reward = reward
        # # TODO; complete the validation trials
        # prev_choice = choice[0]
        # for index, each in enumerate(choice):
        #     if each[0] in [1,2]:
        #         prev_choice = each
        #     else:
        #         choice[index] = prev_choice
        # self.choice = choice
        # self.reward = reward
        return choice, reward

    def _getBlockRewrdProbability(self):
        first_block_reward = np.tile([[0.8, 0.2]], (self.block_size, 1))
        second_block_reward = np.tile([[0.2, 0.8]], (self.block_size, 1))
        reward_prob = np.vstack((first_block_reward, second_block_reward))
        return reward_prob.T

    def _getObjectiveHighest(self, reward_prob):
        objective_highest = []
        trial_num = reward_prob.shape[1]
        for i in range(trial_num):
            trial_reward = reward_prob[:, i]
            max_ind = np.argwhere(trial_reward == np.amax(trial_reward))
            objective_highest.append([max_ind[0, 0], trial_reward[max_ind[0, 0]]])
        return np.array(objective_highest)

    def _getExperiencedRewardProb(self):
        # # TODO: poor result; change to use moving window (20 trials)
        # extract the stimulus choices of all the trials
        self._getChoiceAndReward()
        trial_choice = self.choice
        trial_num = trial_choice.shape[0]
        block_num = trial_num // (self.block_size * 2)
        # Experienced reward probability os a (number of trials in one block, number of blocks) matrix
        choice_reward_prob = np.zeros((self.block_size * 2, block_num))
        for index, choice in enumerate(trial_choice):
            index_in_block = index % (2*self.block_size)
            block_index = index // (2*self.block_size)
            if block_index >= block_num:
                break
            choice_reward_prob[index_in_block, block_index] = self.block_reward_prob[choice[0].item()-1, index_in_block]
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

    def correctRate(self, type='reverse'):
        choice, reward = self._getChoiceAndReward()
        large_blk_size = 2 * self.block_size
        block_num = len(choice) // large_blk_size
        count = np.zeros((block_num, large_blk_size))
        try:
            for step, trial in enumerate(choice):
                blk_index = step // large_blk_size
                count[blk_index, step % large_blk_size] = trial.item()
        except:
            print()
        count = count.astype(int)
        I = np.ones((block_num, self.block_size))
        c = np.hstack((1 * I, 2 * I))
        match = np.array(count == c).astype(int)
        prob = np.mean(match, axis=0)
        plt.plot(np.arange(0, large_blk_size, 1), prob)
        plt.yticks(np.arange(0, 1, 0.1))
        plt.xlabel('trial number')
        plt.ylabel('correct rate')
        plt.show()


if __name__ == '__main__':
    # analyzer = TaskAnalyzer('validate_record-three-armed-2019_12_05-fixed.hdf5')
    analyzer = TaskAnalyzer('../Original-Two-Step-Task/20191231_1001-smp_ts.hdf5')
    analyzer.behaviorAnalysis()
    # analyzer.influenceAnalysis()
    analyzer.correctRate()
