'''
ThreeArmedDataGenerator.py: Generate trainign and validating dataset for three-armed bandit task.

    ----------------------------------------- FOR TRAINING DATA --------------------------------------------
    Generate synthetic trials for the three-armed bandit task. Each trial is represented by a matrix with 
    shape of (number of inputs, number of time steps). Specifically, there are 10 binary-valued inputs:
                0 -- see stimulus A;
                1 -- see stimulus B;
                2 -- see stimulus C;
                3 -- see nothing;
                4 -- choose A;
                5 -- choose B;
                6 -- choose C;
                7 -- do nothing;
                8 -- reward;
                9 -- no reward.
    Each trial is generated over 14 time steps:
                0, 1 -- nothing on the screen; nothing to do
                2, 3, 4 -- show three stimulus;
                5, 6 -- choose stimulus, show three stimulus
                7, 8 -- waiting for reward, only show the chosen stimulus
                9, 10, 11 -- show reward, show the chose stimulus
                12, 13 -- clear the screen,, wait for the next trial.
    ----------------------------------------- FOR TESTING DATA ---------------------------------------------


Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Nov. 25 2019

Reference:
    Separable Learning Systems in the Macaque Brain and the Role of Orbitofrontal Cortex in Contingent Learning
    <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3566584/>
'''

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import scipy.io as sio

class DataGenerate:
    '''
    Description: 
        Generate training set and validating set of three-armed bandit task described in the paper
        ``Separable Learning Systems in the Macaque Brain and the Role of Orbitofrontal Cortex in Contingent Learning'' 
        <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3566584/>.
    
    Variables:
        block_size: The size of a trial, is 150.
        input_dim: The number of features of a trial, is 10.
        time_step_num: The number of time steps for a trial, is 14.
         block_num: The number of blocks after completing missing trials.
         numTrials: The number of trials.
         reward_probability: A matrix of reward probability for each stimulus at each trial, with shape of (3, numTrials).
         choices: A vector of choices of all trials with shape of (1, numTrials).
         rewards: A vector of rewards of all trials with shape of (1, numTrials).
        training_set: Training data set with shape of (numTrials, input_dim, time_step_num).
        reward_type: The type of trend of reward probabilities, either 'fixed' or 'reverse'.
        
    Functions:
        __init__: Initialize the training dataset generator.
        _generateRewardProbability: Compute the reward probability for each trial.
        generating: Generate training dataset.
        save2Mat: Save training data to a .mat file.
    
    '''

    def __init__(self, train_trial_num = 5, validate_trial_num = 5, block_size = 150):
        '''
        Initialize the training data generator.
        :param numTrials: The number of trials to be generated.
        '''
        self.block_size = block_size  # The numebr of trials in a block
        self.input_dim = 10  # The number of features of a trial
        self.time_step_num = 14  # The number of time steps of a trial
        self.validate_time_step_num = 5 # The number of time steps for validating trial,
                                        # only first 5 time steps because we only need validating trial to show stimulus
        self.train_trial_num = train_trial_num
        self.validate_trial_num = validate_trial_num

    def generating(self, reward_type = 'reverse'):
        '''
        Generate training and validating dataset.
        :param reward_type: The type of reward probability, either 'fixed' or 'reverse'.
        :return: VOID
        '''
        # =============== COMPUTE TRIAL REWARD PROBABILITY ========
        self.training_set = []
        self.validating_set = []
        self.reward_probability = self._generateRewardProbability(reward_type)
        print('Finished generating reward probability at each trial.')
        # =============== FOR TRAINING DATA =======================
        # Add some trials to make a complete block in the tail.
        self.train_block_num = self.train_trial_num // self.block_size
        self.train_block_num = (
        self.train_block_num + 1) if self.train_trial_num % self.block_size != 0 else self.train_block_num  # complete the last block
        if self.train_block_num % 2 == 1:  # make the number of blocks be a even number
            self.train_block_num = self.train_block_num + 1
        self.train_trial_num = self.train_block_num * self.block_size
        self.train_trial_num += 1  # TODO: explain
        # =============== FOR VALIDATING DATA ==================
        # Add some trials to make a complete block in the tail.
        self.validate_block_num = self.validate_trial_num // self.block_size
        self.validate_block_num = (
            self.validate_block_num + 1) if self.validate_trial_num % self.block_size != 0 else self.validate_block_num  # complete the last block
        if self.validate_block_num % 2 == 1:  # make the number of blocks be a even number
            self.validate_block_num = self.validate_block_num + 1
        self.validate_trial_num = self.validate_block_num * self.block_size
        # =============== GENERATING TRIALS =======================
        self._generateTraining()
        print('Finished generating {} training data.'.format(self.train_trial_num))
        self._generateValidating()
        print('Finished generating {} validating data.'.format(self.validate_trial_num))

    def _generateRewardProbability(self, reward_type):
        '''
        Compute the reward probability for each trial.
        :param reward_type: The type of reward probabiliy, either 'fixed' or 'reverse'.
        :return: A matrix with shape of (3, numTrials).
        '''
        # return a (3, 2 * block size) matrix
        if 'fixed' == reward_type:
            # The reward probability is basically fixed among trials, with some noise added.
            reward_probability = [[0.8], [0.5], [0.1]]  # reward probability is A (0.8), B (0.5), and C (0.1)
            reward_probability = np.tile(reward_probability, 2*self.block_size)
            reward_probability += np.random.uniform(-0.1, 0.1, (3, 2*self.block_size))
        elif 'reverse' == reward_type:
            # For the trials of the first block, reward probability of A varies based on 0.6,
            # of B varies based on 0.2, and of C is fixed to 0. It is a stable reverse, which means though the
            # perturbation exists, the reward probability of A is no less than B. The reversing process is completed
            # slowly.
            first_base = [[0.6], [0.2], [0.0]]
            first_block = np.tile(first_base, self.block_size-10)
            first_block[0:2,:] = first_block[0:2,:] + np.random.uniform(-0.15, 0.15, (2, self.block_size - 10))
            transit_first_part = [np.linspace(start = first_block[0][-1], stop = 0.4, num = 10),
                            np.linspace(start = first_block[1][-1], stop = 0.4, num = 10),
                            np.linspace(start = first_block[2][-1], stop = 0.4, num=10)]
            transit_first_part = np.array(transit_first_part)
            transit_second_part = [np.linspace(start=transit_first_part[0][-1], stop=0.2, num=20),
                                  np.linspace(start=transit_first_part[1][-1], stop=0.4, num=20),
                                  np.linspace(start=transit_first_part[2][-1], stop=0.8, num=20)]
            transit_second_part = np.array(transit_second_part)
            transit_second_part[:,0:20] = transit_second_part[0:20,:] + np.random.uniform(-0.05, 0.05, (3, 20))
            # For the trials of the second block
            second_base = [[transit_second_part[0][-1]], [transit_second_part[1][-1]], [transit_second_part[2][ -1]]]
            second_block = np.tile(second_base, self.block_size-20)
            second_block = second_block + np.random.uniform(-0.1, 0.1, (3, self.block_size - 20))
            reward_probability = np.concatenate((first_block, transit_first_part, transit_second_part, second_block), axis = 1)
        elif 'sudden_reverse' == reward_type:
            # For the trials of the first block, reward probability of A varies based on 0.6,
            # of B varies based on 0.2, and of C is fixed to 0. It is a stable reverse, which means though the
            # perturbation exists, the reward probability of A is no less than B. Moreover, the reversing process completed
            # suddenly.
            first_base = [[0.6], [0.2], [0.0]]
            first_block = np.tile(first_base, self.block_size)
            first_block[0:2,:] = first_block[0:2,:] + np.random.uniform(-0.15, 0.15, (2, self.block_size))
            # For the trials of the second block
            second_base = [[0.2], [0.4], [0.8]]
            second_block = np.tile(second_base, self.block_size)
            second_block = second_block + np.random.uniform(-0.1, 0.1, (3, self.block_size))
            reward_probability = np.concatenate((first_block, second_block), axis = 1)
        elif 'small_block_reverse' == reward_type:
            # For the trials of the first block, reward probability of A varies based on 0.6,
            # of B varies based on 0.2, and of C is fixed to 0. It is a stable reverse, which means though the
            # perturbation exists, the reward probability of A is no less than B. The reversing process is completed
            # slowly and the block size is 50.
            self.block_size = 20
            first_base = [[0.6], [0.2], [0.0]]
            first_block = np.tile(first_base, self.block_size - 5)
            first_block[0:2, :] = first_block[0:2, :] + np.random.uniform(-0.1, 0.1, (2, self.block_size - 5))
            transit_first_part = [np.linspace(start=first_block[0][-1], stop=0.4, num=5),
                                  np.linspace(start=first_block[1][-1], stop=0.4, num=5),
                                  np.linspace(start=first_block[2][-1], stop=0.4, num=5)]
            transit_first_part = np.array(transit_first_part)
            transit_second_part = [np.linspace(start=transit_first_part[0][-1], stop=0.2, num=10),
                                   np.linspace(start=transit_first_part[1][-1], stop=0.4, num=10),
                                   np.linspace(start=transit_first_part[2][-1], stop=0.8, num=10)]
            transit_second_part = np.array(transit_second_part)
            transit_second_part[:, 0:10] = transit_second_part[0:10, :] + np.random.uniform(-0.05, 0.05, (3, 10))
            # For the trials of the second block
            second_base = [[transit_second_part[0][-1]], [transit_second_part[1][-1]], [transit_second_part[2][-1]]]
            second_block = np.tile(second_base, self.block_size - 10)
            second_block = second_block + np.random.uniform(-0.1, 0.1, (3, self.block_size - 10))
            reward_probability = np.concatenate((first_block, transit_first_part, transit_second_part, second_block),
                                                axis=1)
        elif 'two_reverse' == reward_type:
            # For the trials of the first block, reward probability of A varies based on 0.6,
            # of B varies based on 0.2, and of C is fixed to 0. It is a stable reverse, which means though the
            # perturbation exists, the reward probability of A is no less than B. Then, in the second block, the reward
            # probability reversed to A=0.0, B=0.4, and C=0.8. Moreover, the reversing process completed suddenly.
            first_base = [[0.8], [0.4], [0.0]]
            first_block = np.tile(first_base, self.block_size)
            first_block[0:2, :] = first_block[0:2, :] + np.random.uniform(-0.1, 0.1, (2, self.block_size))
            # For the trials of the second block
            second_base = [[0.0], [0.4], [0.8]]
            second_block = np.tile(second_base, self.block_size)
            second_block[1:,:] = second_block[1:,:] + np.random.uniform(-0.1, 0.1, (2, self.block_size))
            reward_probability = np.concatenate((first_block, second_block), axis=1)
        elif 'two_armed' == reward_type:
            # For the trials of the first block, reward probability of A varies based on 0.6,
            # of B varies based on 0.2, and of C is fixed to 0. It is a stable reverse, which means though the
            # perturbation exists, the reward probability of A is no less than B. Moreover, the reversing process completed
            # suddenly.
            first_base = [[0.6], [0.2], [0.0]]
            first_block = np.tile(first_base, self.block_size)
            first_block[0:2, :] = first_block[0:2, :] + np.random.uniform(-0.1, 0.1, (2, self.block_size))
            # For the trials of the second block
            second_base = [[0.2], [0.6], [0.0]]
            second_block = np.tile(second_base, self.block_size)
            second_block[0:2, :] = second_block[0:2, :] + np.random.uniform(-0.1, 0.1, (2, self.block_size))
            reward_probability = np.concatenate((first_block, second_block), axis=1)
        elif 'two_armed_without_noise' == reward_type:
            # For the trials of the first block, reward probability of A varies based on 0.6,
            # of B varies based on 0.2, and of C is fixed to 0. It is a stable reverse, which means though the
            # perturbation exists, the reward probability of A is no less than B. Moreover, the reversing process completed
            # suddenly.
            first_base = [[0.8], [0.2], [0.0]]
            first_block = np.tile(first_base, self.block_size)
            # For the trials of the second block
            second_base = [[0.2], [0.8], [0.0]]
            second_block = np.tile(second_base, self.block_size)
            reward_probability = np.concatenate((first_block, second_block), axis=1)
        else:
            raise ValueError('Unsupported reward probability type!')

        # show for test
        # plt.plot(np.arange(0, self.block_size * 2), reward_probability[0, :], 'o-r', label='stimulus A')
        # plt.plot(np.arange(0, self.block_size * 2), reward_probability[1, :], 'o-b', label='stimulus B')
        # plt.plot(np.arange(0, self.block_size * 2), reward_probability[2, :], 'o-g', label='stimulus C')
        # plt.yticks(np.arange(0, 1, 0.1))
        # plt.legend(fontsize=20)
        # plt.show()

        self.reward_type = reward_type
        return reward_probability

    def _generateTraining(self):
        '''
        Generate training datasets.
        :return: VOID
        '''
        # ================ GENERATE TRIALS ===================
        self.choices = []  # the choice of stimulus of all the trials, 0 for A, 1 for B, and 2 for C
        self.rewards = []  # the reward of all the trials, 1 for reward and 0 for no reward
        prev_trial = np.zeros((self.input_dim, self.time_step_num))
        for nTrial in range(self.train_trial_num):
            trial = np.zeros(
                (self.input_dim, self.time_step_num + 1))  # add an extra time step for computing loss when training
            # At 0 and 1 time steps, see nothing and do nothing
            trial[3, 0:2] = trial[7, 0:2] = 1
            # At 2, 3, and 4 time steps, see three stimulus and do nothing
            trial[0:3, 2:5] = trial[7, 2:5] = 1
            # At 5 and 6 time steps, see three stimulus always and choose a stimulus
            trial[0:3, 5:7] = 1
            chosen_stimulus = np.random.choice([0, 1, 2], 1)[0]  # choose one from three stimulus: 0 for A, 1 for B, and 2 for C
            self.choices.append(chosen_stimulus)
            trial[chosen_stimulus + 4, 5:7] = 1
            # At 7 and 8 time steps, see three stimulus and wait for reward
            trial[chosen_stimulus, 7:9] = trial[7, 7:9] = 1
            # At 9, 10 and 11 time steps, show the chosen stimulus, show reward, and do nothing
            trial[chosen_stimulus, 9:12] = 1
            trial[7, 9:12] = 1
            reward_prob = self.reward_probability[chosen_stimulus, nTrial % (2*self.block_size)]
            is_rewarded = np.random.choice([0, 1], 1, p=[reward_prob, 1 - reward_prob])[
                0]  # 0 for reward and 1 for no reward
            self.rewards.append(1 - is_rewarded)
            trial[8 + is_rewarded, 9:12] = 1
            # At 12 and 13 time steps, see nothing and do nothing
            trial[3, 12:14] = trial[7, 12:14] = 1
            # The extra time step should be the first step at next trial
            trial[3, 14] = trial[7, 14] = 1
            if nTrial>0:
                # Append this trial into training set
                self.training_set.append(np.hstack((prev_trial, trial)).T)
            prev_trial = trial
        # store weight coefficients for each input of each trial
        self.training_guide = np.vstack((self.rewards[1:], np.tile(2*self.time_step_num, self.train_trial_num-1))).T
        # # ================ SHOW TRIAL ===================
        # sbn.set(font_scale=1.6)
        # y_lables = ['see A', 'see B', 'see C', 'see nothing', 'choose A', 'choose B', 'choose C', 'do nothing',
        #             'reward', 'no reward']
        # show_test = self.training_set[0].T
        # sbn.heatmap(show_test[:, 0:14], cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
        # plt.show()
        # print()

    def _generateValidating(self):
        '''
        Generate validating dataset. 
        Each validating trial only show stimulus, hence, input of first 5 time steps are generated. 
        :return: VOID
        '''
        # ================ GENERATE TRIALS ===================
        trial = np.array([
            [0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 1., 0.],
            [1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0.]
        ])
        for nTrial in range(self.validate_trial_num):
            self.validating_set.append(trial.T)
        # # ================ SHOW TRIAL ===================
        # sbn.set(font_scale=1.6)
        # y_lables = ['see A', 'see B', 'see C', 'see nothing', 'choose A', 'choose B', 'choose C', 'do nothing',
        #             'reward', 'no reward']
        # sbn.heatmap(trial, cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
        # plt.show()
        # print()

    def save2Mat(self):
        '''
        Save data to a ``.mat'' file.
        :return: VOID
        '''
        info = {'NumTrials': self.train_trial_num, 'reward_probability': self.reward_probability,
                'block_size': self.block_size,'input_dim': self.input_dim, 'time_step_num': self.time_step_num}
        pathname = "./data/"
        file_name = datetime.datetime.now().strftime("%Y_%m_%d") + '-blk{}'.format(self.block_size)
        # ================ SAVE TRAINING SET ===================
        train_data_name = 'ThreeArmedBandit_TrainingSet-' + self.reward_type + "-" + file_name
        n = 0
        while 1:
            n += 1
            if not os.path.isfile(pathname + train_data_name + '-' + str(n) + '.mat'):
                sio.savemat(pathname + train_data_name + '-' + str(n) + '.mat',
                            {'training_set': self.training_set,
                             'info':info,
                             'training_guide':self.training_guide,
                             'training_choices':self.choices, # choice of each training trial
                             'training_rewards':self.rewards  # reward of each training trial
                             })
                print("_" * 36)
                print("Training data for three-armed bandit task is saved")
                print("File name:" + pathname + train_data_name + '-' + str(n) + '.mat')
                break

        # ================ SAVE VALIDATING SET ===================
        blkNum = len(self.validating_set) // (2 * self.block_size) + 1
        self.reward_probability = np.tile(self.reward_probability, blkNum)[:, :len(self.validating_set)]
        info = {'NumTrials': self.validate_trial_num, 'reward_probability': self.reward_probability,
                'block_size': self.block_size, 'input_dim': self.input_dim, 'time_step_num': self.time_step_num}
        validate_data_name = 'ThreeArmedBandit_TestingSet-' + self.reward_type + "-" + file_name
        n = 0
        while 1:
            n += 1
            if not os.path.isfile(pathname + validate_data_name + '-' + str(n) + '.mat'):
                sio.savemat(pathname + validate_data_name + '-' + str(n) + '.mat',
                            {'validating_set': self.validating_set,
                              'info': info})
                print("_" * 36)
                print("Validating data for simplified two step task is saved")
                print("File name:" + pathname + validate_data_name + '-' + str(n) + '.mat')
                break

        # # validate the output file
        # filename = pathname + validate_data_name + '-' + str(n) + '.mat'
        # data = sio.loadmat(filename)
        # brief = data['training_descrip'][0, 0]
        # print(brief['info'])


if __name__ == '__main__':
    g = DataGenerate(train_trial_num=100, validate_trial_num= 5000, block_size=50)
    g.generating('sudden_reverse')
    g.save2Mat()
