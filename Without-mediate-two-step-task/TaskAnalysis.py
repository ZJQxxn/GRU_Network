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

    def _getChoiceAndReward(self):
        choice = np.array(self.logFile['choice'])

        count  = 0
        total_count = 0
        for each in choice:
            if each.item() == 1 or each.item() == 2:
                total_count += 1
            if each.item() == 2:
                count += 1
        ratio = count / total_count

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
        plt.xticks(fontsize=20)
        plt.yticks(np.arange(0, 1, 0.1), fontsize=20)
        plt.xlabel('trial', fontsize=20)
        plt.ylabel('choice correct rate', fontsize=20)
        plt.title('[validation trials = 5e3, validation blk size = 70, '
                  'training trials = 7.5e5, training blk size = 50]', fontsize = 20)
        plt.show()


if __name__ == '__main__':
    # analyzer = TaskAnalyzer('validate_record-three-armed-2019_12_05-fixed.hdf5')
    analyzer = TaskAnalyzer('./MyCode-validation-two_step_without_intermediate-revise_interrupt.hdf5')
    # analyzer.behaviorAnalysis()
    # analyzer.influenceAnalysis()
    analyzer.correctRate()
