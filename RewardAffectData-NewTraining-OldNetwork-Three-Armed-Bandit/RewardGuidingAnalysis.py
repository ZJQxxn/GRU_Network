'''
Description:
    Analyze the reward guiding for each block of training trials.
    
Author: 
    Jiaqi Zhang <zjqseu@gmail.com>

Date: 
    May. 4 2020
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

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
    # # Plot for test
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


def generateTraining(block_size = 50, reverse_block_size = 0, need_noise = False):
    '''
    Generate training data. The content of this function is wrote by Zhewei Zhang.
    :param filename: Name of the .mat file, where you want to save the training dataset. 
    :return: VOID
    '''
    NumTrials = int(1e6 + 2)
    reward_prob = np.array([0.8, 0.5, 0.2]).reshape((3, 1)) # TODO: take this as an argument of "_generate..."
    # Reward probability for each trial
    blk_num = NumTrials // (2*block_size) + 1
    # whole_block_reward_prob =  _generateRewardProb(block_size, reverse_block_size, need_noise)
    # whole_block_reward_prob =  _generateHigherBRewardProb(block_size, reverse_block_size, need_noise)
    whole_block_reward_prob =  _generateRealSlowRewardProb(block_size, reverse_block_size, need_noise)

    all_reward_prob = np.tile(whole_block_reward_prob, blk_num)[:, :NumTrials]

    # show trial reward probability
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

    trial_length = 10
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
        # Determine reward
        reward = np.random.uniform(size=1) < all_reward_prob[choice, nTrial]
        reward_all.append(reward)

    next_reward = np.array(reward_all[1:-1]).squeeze()
    next_next_reward = np.array(reward_all[2:]).squeeze()
    training_guide = [1 if next_reward[index] and next_next_reward[index] else 0 for index in range(len(next_reward))]
    reward_all = np.array(reward_all).squeeze()
    return np.array(training_guide).squeeze(), np.array(choices).squeeze(), reward_all, whole_block_reward_prob


def randomGenerateTraining(block_size = 50, reverse_block_size = 0, need_noise = False):
    '''
    Generate training data. The content of this function is wrote by Zhewei Zhang.
    :param filename: Name of the .mat file, where you want to save the training dataset. 
    :return: VOID
    '''
    NumTrials = int(1e6 + 1)
    reward_prob = np.array([0.8, 0.5, 0.2]).reshape((3, 1)) # TODO: take this as an argument of "_generate..."
    # Reward probability for each trial
    blk_num = NumTrials // (2*block_size) + 1
    # whole_block_reward_prob =  _generateRewardProb(block_size, reverse_block_size, need_noise)
    # whole_block_reward_prob =  _generateHigherBRewardProb(block_size, reverse_block_size, need_noise)
    whole_block_reward_prob =  _generateRealSlowRewardProb(block_size, reverse_block_size, need_noise)

    all_reward_prob = np.tile(whole_block_reward_prob, blk_num)[:, :NumTrials]


    n_input = 10
    trial_length = 10
    choices = np.random.choice([0, 1, 2], NumTrials) # 0 for A and 1 for B and 2 for C
    reward_all = []
    for nTrial in range(NumTrials):
        reward = np.random.uniform(size=1) < all_reward_prob[choices[nTrial], nTrial]
        reward_all.append(reward)
    training_guide = np.array(reward_all[1:]).squeeze()
    return np.array(training_guide).squeeze(), np.array(choices).squeeze(), reward_all, whole_block_reward_prob




if __name__ == '__main__':
    train_blk_size = 50
    whole_blk_size = 2 * train_blk_size
    reverse_blk_size = 5
    need_noise = True
    training_guide, choices, rewards, whole_block_reward_prob = generateTraining(block_size = train_blk_size,
                                                       reverse_block_size = reverse_blk_size,
                                                       need_noise = need_noise)
    # the number of trials
    numTrials = len(choices)
    numBlks = numTrials // whole_blk_size if (numTrials // whole_blk_size) % 2 == 0 else (numTrials // whole_blk_size) - 1  # number of small blocks
    numTrials = numBlks * whole_blk_size
    choices = choices[:numTrials]
    training_guide = training_guide[:numTrials]
    print("RAN2 training rate:", np.sum(training_guide) / len(training_guide))

    # random generate training choices
    training_guide, choices, rewards, whole_block_reward_prob = randomGenerateTraining(block_size=train_blk_size,
                                                                                 reverse_block_size=reverse_blk_size,
                                                                                 need_noise=need_noise)
    training_guide = training_guide[:numTrials]
    print("Random training rate:", np.sum(training_guide) / len(training_guide))
    # training_guide = training_guide.reshape((numBlks, -1))
    # training_rate = np.nanmean(training_guide, axis=0)
    # plt.plot(training_rate)
    # plt.show()
    #
    # random_chocies = np.random.choice([0, 1, 2], int(1e6+1))
    #
    #
    # best_choice = np.tile(np.argmax(whole_block_reward_prob, axis = 0).reshape((1,-1)).T, numBlks).T
    # choices = choices.reshape((numBlks, -1))
    # choice_match = (choices == best_choice)
    # best_choice_rate = np.nanmean(choice_match, axis=0)
    #
    # plt.clf()
    # plt.plot(np.nanmean(training_guide == choice_match, axis = 0))
    # plt.show()
