'''
TwoArmedDataGenerator.py: Generate trainign and validating dataset for three-armed bandit task.

    ----------------------------------------- FOR TRAINING DATA --------------------------------------------
    Generate synthetic trials for the three-armed bandit task. Each trial is represented by a matrix with 
    shape of (number of inputs, number of time steps). Specifically, there are 10 binary-valued inputs:
                0 -- see stimulus A;
                1 -- see stimulus B;
                2 -- see nothing;
                3 -- choose A;
                4 -- choose B;
                5 -- do nothing;
                6 -- reward;
                7 -- no reward.
    Each trial is generated over 14 time steps:
                0, 1 -- nothing on the screen; nothing to do
                2, 3, 4 -- show two stimulus;
                5, 6 -- choose stimulus, show two stimulus
                7, 8 -- waiting for reward, only show the chosen stimulus
                9, 10, 11 -- show reward, show the chose stimulus
                12, 13 -- clear the screen,, wait for the next trial.
    ----------------------------------------- FOR TESTING DATA ---------------------------------------------


Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Dec. 16 2019

'''

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import scipy.io as sio
import copy


def generateTraining(filename):
    '''
    Generate training datasets.
    :return: VOID
    '''

    NumTrials = int(7.5e2 + 1)
    reward_prob = np.array([0.8, 0.2]).reshape((2, 1))
    block_size = 50

    # Reward probabiltiy for each trial
    blk_num = NumTrials // (2 * block_size) + 1
    whole_block_reward_prob = np.hstack(
        (
            np.tile(reward_prob, block_size),
            np.tile(1 - reward_prob, block_size)
        ))
    all_reward_prob = np.tile(whole_block_reward_prob, blk_num)[:, :NumTrials]

    # ================ GENERATE TRIALS ===================
    data_ST = []
    n_input = 8
    trial_length = 14
    choices = np.random.choice([0, 1], NumTrials)  # 0 for A and 1 for B
    reward_all = []
    for nTrial in range(NumTrials):
        inputs = np.zeros((n_input, trial_length))
        # show stimulus
        inputs[0:2, 2:5] = 1
        # make choice
        inputs[3 + choices[nTrial], 5:7] = 1
        inputs[0:2, 5:7] = 1
        # show chosen stimulus
        inputs[0 + choices[nTrial], 7:9] = 1
        # show reward
        reward = np.random.uniform(size=1) < all_reward_prob[choices[nTrial], nTrial]
        reward_all.append(reward)
        if reward == 1:
            inputs[6, 9:12] = 1

        inputs[2, :] = inputs[0:2, :].sum(axis=0)
        inputs[2, np.where(inputs[2, :] != 0)] = 1
        inputs[2, :] = 1 - inputs[2, :]
        inputs[5, :] = 1 - inputs[3:5, :].sum(axis=0)
        inputs[7, :] = 1 - inputs[6, :]
        if nTrial != 0:
            data_ST.append([np.hstack((inputs_prev, inputs)).T])
        inputs_prev = copy.deepcopy(inputs)

        # # show trial
        # sbn.set(font_scale=1.6)
        # y_lables = ['show A', 'show B', 'see nothing', 'choose A',
        #             'choose B', 'do nothing', 'reward', 'no reward']
        # sbn.heatmap(inputs, cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
        # plt.show()
        # print()

    training_guide = np.vstack((np.array(reward_all[1:]).squeeze(), np.tile(2 * trial_length, NumTrials - 1))).T

    info = {'NumTrials': NumTrials, 'reward_prob': reward_prob, 'block_size': block_size}
    data_ST_Brief = {'choices': choices, 'reward': reward_all,
                     'training_guide': training_guide}

    n = 0
    while 1:  # save the model
        n += 1
        if not os.path.isfile(pathname + filename + '-' + str(n) + '.mat'):
            sio.savemat(pathname + filename + '-' + str(n) + '.mat',
                        {'data_ST': data_ST,
                         'data_ST_Brief': data_ST_Brief,
                         'info': info})
            print("_" * 36)
            print("training file for simplified two step task is saved")
            print("file name:" + pathname + filename + '-' + str(n) + '.mat')
            break


def generateTesting(filename):
    '''
        Generate testing data. The content of this function is wrote by Zhewei Zhang.
        :param filename: Name of the .mat file, where you want to save the testing dataset. 
        :return: VOID
        '''
    NumTrials = 5000
    reward_prob = np.array([0.8, 0.2]).reshape((2, 1))
    block_size = 70

    # Reward probabiltiy for each trial
    blk_num = NumTrials // (2 * block_size) + 1
    whole_block_reward_prob = np.hstack(
        (
            np.tile(reward_prob, block_size),
            np.tile(1 - reward_prob, block_size)
        ))
    all_reward_prob = np.tile(whole_block_reward_prob, blk_num)[:, :NumTrials]

    inputs = [
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0.],
        [1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0.]
    ]

    # # show trial
    # sbn.set(font_scale=1.6)
    # y_lables = ['show A', 'show B', 'see nothing', 'choose A',
    #             'choose B', 'do nothing', 'reward', 'no reward']
    # sbn.heatmap(np.array(inputs), cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
    # plt.show()
    # print()

    data_ST = [[np.array(inputs).T]] * NumTrials
    data_ST_Brief = {'reward_prob_1': all_reward_prob, 'block_size': block_size}

    n = 0
    while 1:  # save the model
        n += 1
        if not os.path.isfile(pathname + filename + '-' + str(n) + '.mat'):
            sio.savemat(pathname + filename + '-' + str(n) + '.mat',
                        {'data_ST': data_ST,
                         'data_ST_Brief': data_ST_Brief,
                         })
            print("_" * 36)
            print("testing file for simplified two step task is saved")
            print("file name:" + pathname + filename + '-' + str(n) + '.mat')
            break


if __name__ == '__main__':
    pathname = "./data/"
    file_name = datetime.datetime.now().strftime("%Y_%m_%d")
    training_file_name = 'TwoArmed_TrainingSet-' + file_name
    testing_file_name = 'TwoArmed_TestingSet-' + file_name
    generateTraining(training_file_name)
    generateTesting(testing_file_name)