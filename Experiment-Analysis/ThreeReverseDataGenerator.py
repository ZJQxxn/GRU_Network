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
from itertools import product
import h5py

import sys
sys.path.append("../ThreeReverse-Three-Armed-Bandit")
from ThreeArmedTask import ThreeArmedTask

def _generateSevereThreeBlockRewardProb(block_size, reverse_block_size, need_noise):
    '''
    Compute the reward probability for each trial.
    :param reward_type: The type of reward probabiliy, either 'fixed' or 'reverse'.
    :return: A matrix with shape of (3, numTrials).
    '''
    # Three blocks
    first_block = np.tile(np.array([0.8, 0.5, 0.2]).reshape((3,1)), block_size)
    second_block = np.tile(np.array([0.5, 0.2, 0.8]).reshape((3,1)), block_size)
    third_bloc = np.tile(np.array([0.2, 0.8, 0.5]).reshape((3,1)), block_size)
    reward_probability = np.hstack((first_block, second_block, third_bloc))
    # Transtion part
    reward_probability[:, block_size-reverse_block_size : block_size+reverse_block_size] = np.linspace(
        start = reward_probability[:, block_size - reverse_block_size],
        stop = reward_probability[:, block_size + reverse_block_size],
        num = 2*reverse_block_size).T
    reward_probability[:, 2*block_size - reverse_block_size : 2*block_size + reverse_block_size] = np.linspace(
        start=reward_probability[:, 2*block_size - reverse_block_size],
        stop=reward_probability[:, 2*block_size + reverse_block_size],
        num = 2*reverse_block_size).T
    if need_noise:
        reward_probability = reward_probability + np.random.uniform(-0.05, 0.05, (3, 3*block_size))
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


def generateTesting(filename, block_size = 50, reverse_block_size = 0, need_noise = False):
    '''
        Generate testing data. The content of this function is wrote by Zhewei Zhang.
        :param filename: Name of the .mat file, where you want to save the testing dataset. 
        :return: VOID
        '''

    NumTrials = 5000
    reward_prob = np.array([0.8, 0.5, 0.2]).reshape((3,1))
    # block_size = 70

    info = {'NumTrials': NumTrials, 'reward_prob': reward_prob, 'block_size': block_size}

    # Reward probabiltiy for each trial
    blk_num = NumTrials // (2 * block_size) + 1
    whole_block_reward_prob =  _generateSevereThreeBlockRewardProb(block_size, reverse_block_size, need_noise)
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
    data_ST_Brief = {'reward_prob_1': all_reward_prob, 'block_size': block_size}

    sio.savemat(pathname + filename+ '.mat',
                {'data_ST': data_ST,
                'data_ST_Brief': data_ST_Brief,
                })


if __name__ == '__main__':
    _generateSevereThreeBlockRewardProb(50, 5, False)
    # Initialization
    pathname = "./reward-profile-data/"
    blk_size_list = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    reverse_blk_size_list = [0, 5, 10, 15, 20]
    need_noise_list = [True, False]
    # Validation for different reward policies
    all_wining_rate = []
    all_complete_rate = []
    for (blk_size, reverse_blk_size, need_noise) in product(blk_size_list, reverse_blk_size_list, need_noise_list):
        # generate testing data
        testing_file_name = "ThreeReverse-blk{}-reverseblk{}-{}".format(
            blk_size, reverse_blk_size, "noise" if need_noise else "no_noise")
        print(testing_file_name)
        generateTesting(testing_file_name, block_size=blk_size, reverse_block_size=reverse_blk_size,
                        need_noise=need_noise)
        print("Finished generating.")
        # load a trained model and do validation
        model = ThreeArmedTask("ThreeArmed_Config.json", "./reward-profile-data/{}.mat".format(testing_file_name))
        model.loadModel('ThreeReverse-NextTwoReward-ThreeArmed-15e6-model.pt', 'ThreeArmed_Config.json')
        winning_rate, complete_rate = model.validate('temp.hdf5')
        all_wining_rate.append(winning_rate)
        all_complete_rate.append(complete_rate)
        print("Finished validation.")
        # extract choices and rewards
        logFile = h5py.File("temp.hdf5", 'r')
        choice = logFile['choice']
        reward = logFile['reward']
        # firing_rate = logFile['neuron']
        summary = np.hstack((choice, reward))
        np.save("./reward-profile-log/{}-behavior.npy".format(testing_file_name), summary)
        # np.save("./reward-profile-log/{}-firing_rate.npy".format(testing_file_name), firing_rate)
        print("Finished extracting behaviors.\n")
        if "temp.hdf5" in os.listdir("./"):
            os.remove("temp.hdf5")
    np.save("./reward-profile-log/ThreeReverse-all-winning-rate.npy", np.array(all_wining_rate))
    np.save("./reward-profile-log/ThreeReverse-all-complete-rate.npy", np.array(all_complete_rate))



