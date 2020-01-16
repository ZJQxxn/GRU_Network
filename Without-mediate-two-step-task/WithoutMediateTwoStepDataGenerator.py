'''
WithoutMediateTwoStepDataGenerator.py: Generate training and testing  datasets for the two-step task without mediate 
                                       outputs B.

Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Nov. 25 2019

============================== TRAINING TRIALS =============================
Generate synthetic trials for the three-armed bandit task. Each trial is represented by a matrix with shape of (number 
of inputs, number of time steps). 

Specifically, there are 8 binary-valued inputs:
                0 -- see stimulus A1;
                1 -- see stimulus A2;
                2 -- see nothing;
                3 -- do nothing;
                4 -- choose A1;
                5 -- choose A2;
                6 -- reward;
                7 -- no reward.
                
Each trial is generated over 13 time steps:
                0, 1 -- nothing on the screen; nothing to do; no reward
                2, 3, 4 -- show two stimulus; do nothing; no reward
                5, 6 -- choose stimulus (A1/A2); see nothing; no reward
                7, 8, 9 -- show chosen stimulus (A1/A2); do nothing; no reward
                10, 11 -- show reward; see nothing; do nothing
                12 -- clear the screen; wait for the next trial; no reward
'''

import os
import copy
import datetime
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import scipy.io as sio


def generateTraining(filename):
    '''
    Generate training data. The content of this function is wrote by Zhewei Zhang.
    :param filename: Name of the .mat file, where you want to save the training dataset. 
    :return: VOID
    '''
    NumTrials = int(7.5e2 + 1)
    trans_prob = 1  # from A1-B1, from A2-B2  # TODO: set to 1, same as two-armed task
    reward_prob = 0.8
    block_size = 50
    Double = True

    info = {'NumTrials': NumTrials, 'reward_prob': reward_prob, 'block_size': block_size, 'trans_prob': trans_prob}

    temp = np.hstack((np.ones((block_size,)), np.zeros((block_size,))))
    blocks = np.tile(temp,
                     int(NumTrials / (block_size * 2)))  # 0: B1 with large reward prob, 1:b2 with large reward prob
    #
    lost_trialsNum = NumTrials - blocks.size
    if lost_trialsNum <= block_size:
        temp = np.ones((lost_trialsNum,))
    else:
        temp = np.hstack((np.ones((block_size,)), np.zeros((lost_trialsNum - block_size,))))
    blocks = np.hstack((blocks, temp))

    choices = np.random.choice([0, 1], NumTrials)  # 0:A1; 1:A2
    trans_probs = trans_prob * choices + (1 - trans_prob) * (
    1 - choices)  # probability of transition to the B2 in stage 2

    temp = np.random.rand(NumTrials, )
    state2 = trans_probs > temp  # 0: B1; 1: B2

    reward_prob_B1 = (1 - reward_prob) * state2 * (1 - blocks) + reward_prob * (1 - state2) * (
    1 - blocks)  # when B1 with larger reward
    reward_prob_B2 = reward_prob * state2 * blocks + (1 - reward_prob) * (
    1 - state2) * blocks  # when B2 with larger reward

    reward_prob_state2 = reward_prob_B1 + reward_prob_B2  # reward probability of the observation in stage 2

    temp1 = np.random.rand(NumTrials, )
    reward_all = reward_prob_state2 > temp1
    state_all = copy.deepcopy(state2) + 1  # 1: B1; 2: B2



    data_ST = []
    n_input = 8
    trial_length = 13
    shape_Dur = 3  # period for shape presentation
    choice_Dur = 2  # period for shape interval
    for nTrial in range(NumTrials):
        inputs = np.zeros((n_input, trial_length))
        inputs[0:2, 2:5] = 1  # the three-five time points representing the first epoch

        if choices[nTrial] == 0:
            inputs[4, 5:7] = 1
        elif choices[nTrial] == 1:
            inputs[5, 5:7] = 1

        if state_all[nTrial] == 1:
            inputs[0, 7:10] = 1
        elif state_all[nTrial] == 2:
            inputs[1, 7:10] = 1

        if reward_all[nTrial] == 1:
            inputs[6, 10:12] = 1

        inputs[2, :] = inputs[0:2, :].sum(axis=0)
        inputs[2, np.where(inputs[2, :] != 0)] = 1
        inputs[2, :] = 1 - inputs[2, :]
        inputs[3, :] = 1 - inputs[4:6, :].sum(axis=0)
        inputs[7, :] = 1 - inputs[6, :]
        if nTrial != 0:
            data_ST.append([np.hstack((inputs_prev, inputs)).T])
        inputs_prev = copy.deepcopy(inputs)

        # show trial
        # sbn.set(font_scale=1.6)
        # y_lables = ['show A1', 'show A2', 'see nothing', 'do nothing','choose A1',
        #             'choose A2', 'reward', 'no reward']
        # sbn.heatmap(inputs, cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
        # plt.show()
        # print()

    if Double:
        training_guide = np.array(
            [reward_all[1:], 2 * trial_length + np.zeros((len(reward_all) - 1,))]).squeeze().astype(np.int).T.tolist()
    else:
        training_guide = np.array([reward_all[1:], trial_length + np.zeros((len(reward_all) - 1,))]).squeeze().astype(
            np.int).T.tolist()

    data_ST_Brief = {'choices': choices, 'state_all': state_all,
                     'reward': reward_all, 'trans_prob': trans_prob,
                     'shape_Dur': shape_Dur, 'choice_Dur': choice_Dur, 'Double': Double,
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
    # Comments by Zhewei Zhang:
    #   5 inputs represent visual stimulus
    #   6th~9th inputs representing the movement/choice
    #   9th-10th inpust denotes the reward states
    data_ST = []
    n_input = 8
    trial_length = 6

    NumTrials = 5000
    trans_prob = 0.8
    reward_prob = 0.8
    block_size = 70

    info = {'NumTrials': NumTrials, 'reward_prob': reward_prob, 'block_size': block_size, 'trans_prob': trans_prob}

    temp = np.hstack((np.ones((block_size,)), np.zeros((block_size,))))
    blocks = np.tile(temp, int(NumTrials / (block_size * 2)))
    #
    lost_trialsNum = NumTrials - blocks.size
    if lost_trialsNum <= block_size:
        temp = np.ones((lost_trialsNum,))
    else:
        temp = np.hstack((np.ones((block_size,)), np.zeros((lost_trialsNum - block_size,))))
    blocks = np.hstack((blocks, temp))

    trans_probs = trans_prob * np.ones(NumTrials, )
    reward_probs = reward_prob * blocks + (1 - reward_prob) * (1 - blocks)

    inputs = [
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0.]
    ]

    # chosen_state = [
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    # ]
    #

    # show trial
    # sbn.set(font_scale=1.6)
    # y_lables = ['show A1', 'show A2', 'see nothing', 'do nothing', 'choose A1',
    #             'choose A2', 'reward', 'no reward']
    # sbn.heatmap(np.array(inputs), cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
    # plt.show()
    # print()

    data_ST = [[np.array(inputs).T]] * NumTrials
    data_ST_Brief = {'reward_prob_1': reward_probs, 'trans_probs': trans_probs, 'block_size': block_size,
                     'block': blocks}

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
    training_file_name = 'WithoutMediateTwo_TrainingSet-' + file_name
    testing_file_name = 'WithoutMediateTwo_TestingSet-' + file_name
    generateTraining(training_file_name)
    generateTesting(testing_file_name)