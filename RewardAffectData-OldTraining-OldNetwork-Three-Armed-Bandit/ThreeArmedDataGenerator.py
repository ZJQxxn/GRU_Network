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

def _generateRewardProb(block_size = 50, reverse_block_size = 0, need_noise = False):
    '''
           Compute the reward probability for each trial.
           :param reward_type: The type of reward probabiliy, either 'fixed' or 'reverse'.
           :return: A matrix with shape of (3, numTrials).
    '''
    first_base = [[0.8], [0.5], [0.2]]
    first_block = np.tile(first_base, block_size - reverse_block_size)
    # The transit part
    if reverse_block_size > 0:
        transit_first_part = [np.linspace(start=first_block[0][-1], stop=first_base[1][0], num=reverse_block_size),
                              np.tile(first_base[1][0], reverse_block_size),
                              np.linspace(start=first_block[2][-1], stop=first_base[1][0], num=reverse_block_size)]
        transit_first_part = np.array(transit_first_part)
        transit_second_part = [
            np.linspace(start=transit_first_part[0][-1], stop=first_base[2][0], num=reverse_block_size),
            np.tile(first_base[1][0], reverse_block_size),
            np.linspace(start=transit_first_part[2][-1], stop=first_base[0][0], num=reverse_block_size)]
        transit_second_part = np.array(transit_second_part)
        transit_part = np.concatenate((transit_first_part, transit_second_part), axis=1)
        second_base = [[transit_part[each][-1]] for each in [0, 1, 2]]
    elif reverse_block_size == 0:
        transit_part = []
        second_base = [[first_block[each][-1]] for each in [2, 1, 0]] # Reverse the reward probability
    else:
        raise ValueError("The reverse block size should be no less than 0!")
    # For the trials of the second block
    second_block = np.tile(second_base, block_size - reverse_block_size)
    # Concatenate each part and add noise, if need any
    if reverse_block_size > 0:
        reward_probability = np.concatenate((first_block, transit_part, second_block), axis=1)
    else:
        reward_probability = np.concatenate((first_block, second_block), axis=1)
    if need_noise:
        reward_probability = reward_probability + np.random.uniform(-0.05, 0.05, (3, 2*block_size))
    # # Show for test
    # plt.figure(figsize=(15,8))
    # plt.title("Objective Reward Probability", fontsize = 20)
    # plt.plot(np.arange(0, block_size * 2), reward_probability[0, :], 'o-r', label='stimulus A')
    # plt.plot(np.arange(0, block_size * 2), reward_probability[1, :], 'o-b', label='stimulus B')
    # plt.plot(np.arange(0, block_size * 2), reward_probability[2, :], 'o-g', label='stimulus C')
    # plt.yticks(np.arange(0, 1.1, 0.2), fontsize = 20)
    # plt.ylabel("Reward Probability", fontsize = 20)
    # plt.xticks(fontsize=20)
    # plt.xlabel("Trial", fontsize=20)
    # plt.legend(loc = 9, fontsize=20, ncol = 3) # legend is located at upper center with 3 columns
    # plt.show()
    return reward_probability

def _generateLowerBRewardProb(block_size = 50, reverse_block_size = 0, need_noise = False):
    '''
           Compute the reward probability for each trial.
           :param reward_type: The type of reward probabiliy, either 'fixed' or 'reverse'.
           :return: A matrix with shape of (3, numTrials).
    '''
    first_base = [[0.8], [0.3], [0.2]]
    first_block = np.tile(first_base, block_size - reverse_block_size)
    # The transit part
    if reverse_block_size > 0:
        transit_first_part = [np.linspace(start=first_block[0][-1], stop=first_base[1][0], num=reverse_block_size),
                              np.tile(first_base[1][0], reverse_block_size),
                              np.linspace(start=first_block[2][-1], stop=first_base[1][0], num=reverse_block_size)]
        transit_first_part = np.array(transit_first_part)
        transit_second_part = [
            np.linspace(start=transit_first_part[0][-1], stop=first_base[2][0], num=reverse_block_size),
            np.tile(first_base[1][0], reverse_block_size),
            np.linspace(start=transit_first_part[2][-1], stop=first_base[0][0], num=reverse_block_size)]
        transit_second_part = np.array(transit_second_part)
        transit_part = np.concatenate((transit_first_part, transit_second_part), axis=1)
        second_base = [[transit_part[each][-1]] for each in [0, 1, 2]]
    elif reverse_block_size == 0:
        transit_part = []
        second_base = [[first_block[each][-1]] for each in [2, 1, 0]] # Reverse the reward probability
    else:
        raise ValueError("The reverse block size should be no less than 0!")
    # For the trials of the second block
    second_block = np.tile(second_base, block_size - reverse_block_size)
    # Concatenate each part and add noise, if need any
    if reverse_block_size > 0:
        reward_probability = np.concatenate((first_block, transit_part, second_block), axis=1)
    else:
        reward_probability = np.concatenate((first_block, second_block), axis=1)
    if need_noise:
        reward_probability = reward_probability + np.random.uniform(-0.05, 0.05, (3, 2*block_size))
    # # Show for test
    # plt.figure(figsize=(15,8))
    # plt.title("Objective Reward Probability", fontsize = 20)
    # plt.plot(np.arange(0, block_size * 2), reward_probability[0, :], 'o-r', label='stimulus A')
    # plt.plot(np.arange(0, block_size * 2), reward_probability[1, :], 'o-b', label='stimulus B')
    # plt.plot(np.arange(0, block_size * 2), reward_probability[2, :], 'o-g', label='stimulus C')
    # plt.yticks(np.arange(0, 1.1, 0.2), fontsize = 20)
    # plt.ylabel("Reward Probability", fontsize = 20)
    # plt.xticks(fontsize=20)
    # plt.xlabel("Trial", fontsize=20)
    # plt.legend(loc = 9, fontsize=20, ncol = 3) # legend is located at upper center with 3 columns
    # plt.show()
    return reward_probability

def _generateHigherBRewardProb(block_size = 50, reverse_block_size = 0, need_noise = False):
    '''
           Compute the reward probability for each trial.
           :param reward_type: The type of reward probabiliy, either 'fixed' or 'reverse'.
           :return: A matrix with shape of (3, numTrials).
    '''
    first_base = [[0.8], [0.6], [0.2]]
    first_block = np.tile(first_base, block_size - reverse_block_size)
    # The transit part
    if reverse_block_size > 0:
        transit_first_part = [np.linspace(start=first_block[0][-1], stop=first_base[1][0], num=reverse_block_size),
                              np.tile(first_base[1][0], reverse_block_size),
                              np.linspace(start=first_block[2][-1], stop=first_base[1][0], num=reverse_block_size)]
        transit_first_part = np.array(transit_first_part)
        transit_second_part = [
            np.linspace(start=transit_first_part[0][-1], stop=first_base[2][0], num=reverse_block_size),
            np.tile(first_base[1][0], reverse_block_size),
            np.linspace(start=transit_first_part[2][-1], stop=first_base[0][0], num=reverse_block_size)]
        transit_second_part = np.array(transit_second_part)
        transit_part = np.concatenate((transit_first_part, transit_second_part), axis=1)
        second_base = [[transit_part[each][-1]] for each in [0, 1, 2]]
    elif reverse_block_size == 0:
        transit_part = []
        second_base = [[first_block[each][-1]] for each in [2, 1, 0]] # Reverse the reward probability
    else:
        raise ValueError("The reverse block size should be no less than 0!")
    # For the trials of the second block
    second_block = np.tile(second_base, block_size - reverse_block_size)
    # Concatenate each part and add noise, if need any
    if reverse_block_size > 0:
        reward_probability = np.concatenate((first_block, transit_part, second_block), axis=1)
    else:
        reward_probability = np.concatenate((first_block, second_block), axis=1)
    if need_noise:
        reward_probability = reward_probability + np.random.uniform(-0.05, 0.05, (3, 2*block_size))
    # # Show for test
    # plt.figure(figsize=(15,8))
    # plt.title("Objective Reward Probability", fontsize = 20)
    # plt.plot(np.arange(0, block_size * 2), reward_probability[0, :], 'o-r', label='stimulus A')
    # plt.plot(np.arange(0, block_size * 2), reward_probability[1, :], 'o-b', label='stimulus B')
    # plt.plot(np.arange(0, block_size * 2), reward_probability[2, :], 'o-g', label='stimulus C')
    # plt.yticks(np.arange(0, 1.1, 0.2), fontsize = 20)
    # plt.ylabel("Reward Probability", fontsize = 20)
    # plt.xticks(fontsize=20)
    # plt.xlabel("Trial", fontsize=20)
    # plt.legend(loc = 9, fontsize=20, ncol = 3) # legend is located at upper center with 3 columns
    # plt.show()
    return reward_probability


def _generateRealSlowRewardProb(block_size = 50, reverse_block_size = 0, need_noise = False):
    '''
           Compute the reward probability for each trial.
           :param reward_type: The type of reward probabiliy, either 'fixed' or 'reverse'.
           :return: A matrix with shape of (3, numTrials).
    '''
    first_base = [[0.8], [0.5], [0.2]]
    first_block = np.concatenate(
        (
            [np.linspace(start= 0.5, stop=first_base[0][0], num = reverse_block_size),
            np.linspace(start=0.5, stop=first_base[1][0], num = reverse_block_size),
            np.linspace(start=0.5, stop=first_base[2][0], num = reverse_block_size)],
            np.tile(first_base, block_size - 2 * reverse_block_size)
        ), axis = 1)
    # The transit part
    if reverse_block_size > 0:
        transit_first_part = [np.linspace(start=first_block[0][-1], stop=first_base[1][0], num=reverse_block_size),
                              np.tile(first_block[1][-1], reverse_blk_size),
                              np.linspace(start=first_block[2][-1], stop=first_base[1][0], num=reverse_block_size)]
        transit_first_part = np.array(transit_first_part)
        transit_second_part = [
            np.linspace(start=transit_first_part[0][-1], stop=first_base[2][0], num=reverse_block_size),
            np.tile(transit_first_part[1][-1], reverse_blk_size),
            np.linspace(start=transit_first_part[2][-1], stop=first_base[0][0], num=reverse_block_size)]
        transit_second_part = np.array(transit_second_part)
        transit_part = np.concatenate((transit_first_part, transit_second_part), axis=1)
        second_base = [[transit_part[each][-1]] for each in [0, 1, 2]]
        second_block = np.tile(second_base, block_size - 2 * reverse_block_size)
        last_transit = [np.linspace(start=second_block[0][-1], stop=first_base[1][0], num=reverse_block_size),
                        # stimulus A
                        np.linspace(start=second_block[1][-1], stop=first_base[1][0], num=reverse_block_size),
                        # stimulus B
                        np.linspace(start=second_block[2][-1], stop=first_base[1][0], num=reverse_block_size)]
        second_block = np.concatenate((second_block, last_transit), axis=1)
        reward_probability = np.concatenate((first_block, transit_part, second_block), axis=1)
    elif reverse_block_size == 0:
        second_base = [[first_block[each][-1]] for each in [2, 1, 0]]  # Reverse the reward probability
        second_block = np.tile(second_base, block_size)
        reward_probability = np.concatenate((first_block, second_block), axis=1)
    else:
        raise ValueError("The reverse block size should be no less than 0!")
    if need_noise:
        reward_probability = reward_probability + np.random.uniform(-0.05, 0.05, (3, 2 * block_size))
    # Plot for test
    plt.figure(figsize=(15,8))
    plt.title("Objective Reward Probability", fontsize = 20)
    plt.plot(np.arange(0, block_size * 2), reward_probability[0, :], 'o-r', label='stimulus A')
    plt.plot(np.arange(0, block_size * 2), reward_probability[1, :], 'o-b', label='stimulus B')
    plt.plot(np.arange(0, block_size * 2), reward_probability[2, :], 'o-g', label='stimulus C')
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize = 20)
    plt.ylabel("Reward Probability", fontsize = 20)
    plt.xticks(fontsize=20)
    plt.xlabel("Trial", fontsize=20)
    plt.legend(loc = 9, fontsize=20, ncol = 3) # legend is located at upper center with 3 columns
    plt.show()
    return reward_probability


def generateTraining(filename, block_size = 50, reverse_block_size = 0, need_noise = False):
    '''
    Generate training data. The content of this function is wrote by Zhewei Zhang.
    :param filename: Name of the .mat file, where you want to save the training dataset. 
    :return: VOID
    '''
    NumTrials = int(1e6 + 1)
    reward_prob = np.array([0.8, 0.5, 0.2]).reshape((3, 1)) # TODO: take this as an argument of "_generate..."
    # Reward probability for each trial
    blk_num = NumTrials // (2*block_size) + 1
    whole_block_reward_prob =  _generateRewardProb(block_size, reverse_block_size, need_noise)
    # whole_block_reward_prob =  _generateHigherBRewardProb(block_size, reverse_block_size, need_noise)
    # whole_block_reward_prob =  _generateRealSlowRewardProb(block_size, reverse_block_size, need_noise)

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
    last_choice = None
    last_reward = False
    for nTrial in range(NumTrials):
        # If the last trial is rewarded, repeat the choice
        if not last_choice is None and last_reward:
            choice = last_choice
        else:
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
    whole_block_reward_prob =  _generateRewardProb(block_size, reverse_block_size, need_noise)
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
    train_blk_size = 50
    test_blk_size = 70
    reverse_blk_size = 0
    need_noise = True
    pathname = "./data/"
    training_file_name = datetime.datetime.now().strftime("%Y_%m_%d") \
                + "-blk{}-reverseblk{}-{}".format(train_blk_size, reverse_blk_size, "noise" if need_noise else "no_noise")
    training_file_name = 'RewardAffect_ThreeArmed_TrainingSet-' + training_file_name
    testing_file_name = datetime.datetime.now().strftime("%Y_%m_%d") \
                         + "-blk{}-reverseblk{}-{}".format(test_blk_size, reverse_blk_size,"noise" if need_noise else "no_noise")
    testing_file_name = 'RewardAffect_ThreeArmed_TestingSet-' + testing_file_name
    generateTraining(training_file_name, block_size = train_blk_size, reverse_block_size = reverse_blk_size, need_noise = need_noise)
    generateTesting(testing_file_name, block_size = test_blk_size, reverse_block_size = reverse_blk_size, need_noise = need_noise)

