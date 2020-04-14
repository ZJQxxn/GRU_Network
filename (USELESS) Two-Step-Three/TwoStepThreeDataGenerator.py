'''
TwoStepThreeDataGenerator.py: Generate training and testing  datasets for the two-step task with three stimulus and 
                                three intermediate outputs.

Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Feb. 10 2020

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
    NumTrials = int(5e5 + 1)
    trans_prob = 1  # from A1-B1, from A2-B2, from A3-B3
    first_block_reward_prob = [0.8, 0.5, 0.2]
    second_block_reward_prob = [0.2, 0.5, 0.8]

    block_size = 50
    Double = True

    info = {'NumTrials': NumTrials, 'first_reward_prob': first_block_reward_prob, 'second_reward_prob':second_block_reward_prob,
            'block_size': block_size, 'trans_prob': trans_prob}

    # temp = np.hstack((np.ones((block_size,)), np.zeros((block_size,))))
    # blocks = np.tile(temp,
    #                  int(NumTrials / (block_size * 2)))  # 0: B1 with large reward prob, 1:b2 with large reward prob
    # #
    # lost_trialsNum = NumTrials - blocks.size
    # if lost_trialsNum <= block_size:
    #     temp = np.ones((lost_trialsNum,))
    # else:
    #     temp = np.hstack((np.ones((block_size,)), np.zeros((lost_trialsNum - block_size,))))
    # blocks = np.hstack((blocks, temp)) # in the first block, A1 has the highest probability while in the second block, A3 has the highest prob.

    choices = np.random.choice([0, 1, 2], NumTrials)  # 0:A1; 1:A2; 2:A3
    state2 = choices # 0:B1; 1:B2; 2:B3. Because the tran_prob = 1

    reward_prob = np.zeros((NumTrials,))
    count = 0
    for (index, each) in enumerate(choices):
        if count < 50:
            reward_guide = first_block_reward_prob
        elif 50 <= count < 100:
            reward_guide = second_block_reward_prob
        else:
            reward_guide = first_block_reward_prob
            count = 0
        reward_prob[index] = reward_guide[each]

    reward_all = reward_prob > np.random.rand(NumTrials, )
    state_all = copy.deepcopy(state2) + 1  # 1: B1; 2: B2; 3: B3

    # Comments by Zhewei Zhang:
    #   first seven inputs represent visual stimulus
    #   1st~2nd inputs representing the options
    #   3rd~4th inputs representing the intermeidate outcome
    #   6th~8th inputs representing the movement/choice
    #   9th~10th inpust denotes the reward states

    data_ST = []
    n_input = 13
    trial_length = 13
    shape_Dur = 3  # period for shape presentation
    choice_Dur = 2  # period for shape interval
    for nTrial in range(NumTrials):
        inputs = np.zeros((n_input, trial_length))
        # Show A
        inputs[0:3, 2:5] = 1  # the three-five time points representing the first epoch
        # Choose A
        inputs[8 + choices[nTrial], 5:7] = 1
        # Show B
        inputs[3 + state_all[nTrial]-1, 7:10] = 1
        # Show reward
        if reward_all[nTrial] == 1:
            inputs[11, 10:12] = 1

        inputs[6, :] = inputs[0:6, :].sum(axis=0)
        inputs[6, np.where(inputs[6, :] != 0)] = 1
        inputs[6, :] = 1 - inputs[6, :]
        inputs[7, :] = 1 - inputs[8:11, :].sum(axis=0)
        inputs[12, :] = 1 - inputs[11, :]
        if nTrial != 0:
            data_ST.append([np.hstack((inputs_prev, inputs)).T])
        inputs_prev = copy.deepcopy(inputs)

        # # show trial
        # sbn.set(font_scale=1.6)
        # y_lables = ['show A1', 'show A2', 'show A3', 'show B1', 'show B2', 'show B3', 'see nothing', 'do nothing','choose A1',
        #             'choose A2', 'choose A3', 'reward', 'no reward']
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
    n_input = 13
    trial_length = 6

    NumTrials = 5000
    trans_prob = 1
    first_block_reward_prob = [0.8, 0.5, 0.2]
    second_block_reward_prob = [0.2, 0.5, 0.8]

    block_size = 70

    info = {'NumTrials': NumTrials, 'first_reward_prob': first_block_reward_prob,
            'second_reward_prob': second_block_reward_prob,
            'block_size': block_size, 'trans_prob': trans_prob}

    temp = np.hstack((np.ones((block_size,)), np.zeros((block_size,))))
    blocks = np.tile(temp, int(NumTrials / (block_size * 2)))
    #
    lost_trialsNum = NumTrials - blocks.size
    if lost_trialsNum <= block_size:
        temp = np.ones((lost_trialsNum,))
    else:
        temp = np.hstack((np.ones((block_size,)), np.zeros((lost_trialsNum - block_size,))))
    blocks = np.hstack((blocks, temp))

    one_whole_block_reward = np.hstack((np.tile(np.array(first_block_reward_prob).reshape((3,1)), 70),
                                        np.tile(np.array(second_block_reward_prob).reshape((3, 1)), 70)))
    reward_prob = np.tile(one_whole_block_reward, NumTrials // 140 +1)[:, :NumTrials]

    inputs = [
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 0.]
    ]

    # # show trial
    # sbn.set(font_scale=1.6)
    # y_lables = ['show A1', 'show A2', 'show A3', 'show B1', 'show B2', 'show B3', 'see nothing', 'do nothing',
    #             'choose A1','choose A2','chosse A3', 'reward', 'no reward']
    # sbn.heatmap(np.array(inputs), cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
    # plt.show()
    # print()

    data_ST = [[np.array(inputs).T]] * NumTrials
    data_ST_Brief = {'NumTrials': NumTrials, 'reward_prob':reward_prob,'trans_probs': trans_prob,
                     'block_size': block_size,'block': blocks}

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
    training_file_name = 'TwoStepThree_TrainingSet-' + file_name
    testing_file_name = 'TwoStepThree_TestingSet-' + file_name
    generateTraining(training_file_name)
    # generateTesting(testing_file_name)