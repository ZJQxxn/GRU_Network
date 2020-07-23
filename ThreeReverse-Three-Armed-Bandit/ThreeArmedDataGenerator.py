'''
TwoArmedDataGenerator.py: Generate training and validating dataset for three-armed bandit task.

    ----------------------------------------- FOR TRAINING DATA --------------------------------------------
    Generate synthetic trials for the three-armed bandit task. Each trial is represented by a matrix with 
    shape of (number of inputs, number of time steps). Specifically, there are 10 binary-valued inputs:
                0 -- see stimulus A;
                1 -- see stimulus B;
                2 -- see stimulus C
                3 -- see nothing;
                4 -- do nothing;
                5 -- choose A;
                6 -- choose B;
                7 -- choose C
                8 -- reward;
                9 -- no reward.
    Each trial is generated over 14 time steps:
                0, 1 -- nothing on the screen; nothing to do
                2, 3, 4 -- show two stimulus;
                5, 6 -- choose stimulus, show chosen stimulus
                7, 8 -- show reward
                9 -- nothing on the screen


Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: May. 1 2020

'''
import os
import copy
import datetime
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import scipy.io as sio

def _generateRewardProb():
    #TODO: fix the generating process for now
    '''
           Compute the reward probability for each trial.
           :param reward_type: The type of reward probabiliy, either 'fixed' or 'reverse'.
           :return: A matrix with shape of (3, numTrials).
    '''
    # TODO: reverse among three stimulus, rather than only A and C
    first_block = np.tile(np.array([0.8, 0.5, 0.2]).reshape((3,1)), 50)
    second_block = np.tile(np.array([0.5, 0.8, 0.2]).reshape((3,1)), 50)
    third_bloc = np.tile(np.array([0.2, 0.8, 0.5]).reshape((3,1)), 50)
    fourth_block = np.tile(np.array([0.2, 0.5, 0.8]).reshape((3,1)), 50)
    reward_probability = np.hstack((first_block, second_block, third_bloc, fourth_block))
    # # Plot for test
    # trial_num = reward_probability.shape[1]
    # plt.figure(figsize=(15,8))
    # plt.title("Objective Reward Probability", fontsize = 20)
    # plt.plot(np.arange(0, trial_num), reward_probability[0, :], 'o-r', label='stimulus A')
    # plt.plot(np.arange(0, trial_num), reward_probability[1, :], 'o-b', label='stimulus B')
    # plt.plot(np.arange(0, trial_num), reward_probability[2, :], 'o-g', label='stimulus C')
    # plt.yticks(np.arange(0, 1.1, 0.2), fontsize = 20)
    # plt.ylabel("Reward Probability", fontsize = 20)
    # plt.xticks(fontsize=20)
    # plt.xlabel("Trial", fontsize=20)
    # plt.legend(loc = 9, fontsize=20, ncol = 3) # legend is located at upper center with 3 columns
    # plt.show()
    return reward_probability


def generateTraining(filename):
    '''
    Generate training data. The content of this function is wrote by Zhewei Zhang.
    :param filename: Name of the .mat file, where you want to save the training dataset. 
    :return: VOID
    '''
    NumTrials = int(1e3 + 1)
    reward_prob = np.array([0.8, 0.5, 0.2]).reshape((3, 1))
    # Reward probability for each trial
    whole_block_size = 200
    blk_num = NumTrials // (whole_block_size) + 1
    # whole_block_reward_prob =  _generateRewardProb(block_size, reverse_block_size, need_noise)
    # whole_block_reward_prob =  _generateHigherBRewardProb(block_size, reverse_block_size, need_noise)
    whole_block_reward_prob =  _generateRewardProb()

    all_reward_prob = np.tile(whole_block_reward_prob, blk_num)[:, :NumTrials]

    # # show trial reward probability
    # plt.title('Trial Reward Probability', fontsize=30)
    # plt.plot(np.arange(0, 100), whole_block_reward_prob[0, :], label='Stimulus A')
    # plt.plot(np.arange(0, 100), whole_block_reward_prob[1, :], label='Stimulus B')
    # plt.plot(np.arange(0, 100), whole_block_reward_prob[2, :], label='Stimulus C')
    # plt.xlabel('Trial', fontsize=30)
    # plt.ylabel('Probability', fontsize=30)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    # plt.legend(fontsize=25)
    # plt.show()
    # print()

    # Generat training data
    data_ST = []
    n_input = 10
    trial_length = 10
    # choices = np.random.choice([0, 1, 2], NumTrials) # 0 for A and 1 for B and 2 for C
    choices = []
    reward_all = []
    for nTrial in range(NumTrials):
        choice = np.random.choice([0, 1, 2], 1)
        choices.append(choice)
        last_choice = choice
        # Set input matrix
        inputs = np.zeros((n_input, trial_length))
        # show stimulus
        inputs[0:3, 2:5] = 1
        # make choice
        inputs[5 + choice, 5:7] = 1
        inputs[0 + choice, 5:7] = 1
        # show reward
        reward = np.random.uniform(size=1) < all_reward_prob[choice, nTrial]
        reward_all.append(reward)
        if reward == 1:
            inputs[8, 7:9] = 1
        last_reward = reward
        inputs[3, :] = inputs[0:3, :].sum(axis=0)
        inputs[3, np.where(inputs[3, :] != 0)] = 1
        inputs[3, :] = 1 - inputs[3, :]
        inputs[4, :] = 1 - inputs[5:8, :].sum(axis=0)
        inputs[9, :] = 1 - inputs[8, :]

        if nTrial != 0:
            data_ST.append([np.hstack((inputs_prev, inputs)).T])
        inputs_prev = copy.deepcopy(inputs)

        # # show trial
        # sbn.set(font_scale=1.6)
        # y_lables = ['show A', 'show B', 'show C', 'see nothing', 'do nothing','choose A',
        #             'choose B', 'choose C', 'reward', 'no reward']
        # sbn.heatmap(inputs, cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
        # plt.show()
        # print()

    training_guide = np.vstack((np.array(reward_all[1:]).squeeze(), np.tile(2 * trial_length, NumTrials - 1))).T

    info = {'NumTrials': NumTrials, 'reward_prob': reward_prob, 'block_size': whole_block_size}
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

    whole_block_size = 200
    NumTrials = 5000
    reward_prob = np.array([0.8, 0.5, 0.2]).reshape((3,1))
    # block_size = 70

    info = {'NumTrials': NumTrials, 'reward_prob': reward_prob, 'block_size': whole_block_size}

    # Reward probabiltiy for each trial
    blk_num = NumTrials // (whole_block_size) + 1
    whole_block_reward_prob =  _generateRewardProb()
    all_reward_prob = np.tile(whole_block_reward_prob, blk_num)[:, :NumTrials]

    inputs = [
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0.],
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
    # y_lables = ['show A', 'show B', 'show C', 'see nothing', 'do nothing', 'choose A',
    #             'choose B', 'choose C', 'reward', 'no reward']
    # sbn.heatmap(np.array(inputs), cmap="YlGnBu", linewidths=0.5, yticklabels=y_lables)
    # plt.show()
    # print()

    data_ST = [[np.array(inputs).T]] * NumTrials
    data_ST_Brief = {'reward_prob_1': all_reward_prob, 'block_size': whole_block_size}

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
    training_file_name = 'ThreeReverse_ThreeArmed_TrainingSet-' + datetime.datetime.now().strftime("%Y_%m_%d")
    testing_file_name = 'ThreeReverse_ThreeArmed_TestingSet-' + datetime.datetime.now().strftime("%Y_%m_%d")
    generateTraining(training_file_name)
    generateTesting(testing_file_name)

