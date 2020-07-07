'''
Description:
    Analyze the influence of history-choice and history-reward on the current trial.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    May. 19 2020
'''

import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sbn
import numpy as np
import h5py
import re


def getChoiceAndReward(logFileName, dataFileName):
    '''
    Pre-processing. Handle with nan data.
    :param logFileName: Log filename.
    :param dataFileName: Data filename.
    :return: Processed choices and rewards.
             choices: All choices.
             rewards: All rewards.
    '''
    logFile = h5py.File(logFileName, 'r')
    reward_prob = loadmat(dataFileName)['data_ST_Brief']['reward_prob_1'][0,0]
    print(list(logFile.keys()))
    choices = logFile['choice'].value
    rewards = logFile['reward'].value
    # Handle with nan data
    for index, each in enumerate(choices):
        if np.nan == each or each > 3:
            cur_choice = choices[index - 1]
            if np.random.uniform(0, 1, 1) <= reward_prob[cur_choice - 1, index]:
                cur_reward = 1
            else:
                cur_reward = 0
            choices[index] = cur_choice
            rewards[index] = cur_reward
    return choices, rewards


def historyAnalysisFor2(choices, rewards):
    choices_sequence = ''.join(np.array(choices.squeeze(), dtype=np.str))
    # Find specific patterns
    repeat_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ratio = np.zeros((len(repeat_list), 4))
    for index, repeat_num in enumerate(repeat_list):
        # print("="*10, repeat_num, "="*10)
        res_iter = _findTwoPattern(choices_sequence, repeat_num = repeat_num)
        reward_determin_index = np.array([each[0] for each in res_iter]) + repeat_num
        choice_determin_index = np.array([each[0] for each in res_iter]) + repeat_num + 1
        previous_choice = np.array([each[1] for each in res_iter])
        # Determine if the last choice is rewarded or not
        if_rewarded = rewards[reward_determin_index].squeeze()
        if_keep_choice = (choices[choice_determin_index].squeeze() == previous_choice)
        if_not_keep_choice = (choices[choice_determin_index].squeeze() == (3 - previous_choice))
        # Collect data
        reward_index = np.where(if_rewarded == 1)[0]
        not_reward_index = np.where(if_rewarded == 0)[0]
        ratio[index, :] = [
            np.sum(if_keep_choice[reward_index]) / len(reward_index),
            np.sum(if_keep_choice[not_reward_index]) / len(not_reward_index),
            np.sum(if_not_keep_choice[reward_index]) / len(reward_index),
            np.sum(if_not_keep_choice[not_reward_index]) / len(not_reward_index)
        ]
    return ratio


def plotAnalysisFor2(ratio, task_name):
    # Plot for reward and not reward
    plt.figure(figsize=(16, 12))
    plt.title(task_name, fontsize = 20)
    # Choosing A
    plt.subplot(1, 2, 1)
    plt.plot(ratio[:5, 0], "o-", label="B Rewarded", lw=3, ms=10)
    plt.plot(ratio[:5, 1], "o--", label="B Not Rewarded", lw=3, ms=10)
    plt.xlabel("Choice History", fontsize=20)
    plt.ylabel("Likelihood of Choosing A", fontsize=20)
    plt.ylim(0.2, 0.9)
    plt.yticks(fontsize=20)
    plt.xticks(np.arange(5), ["$A^{}$B".format(each) for each in range(1, 6)], fontsize=20)
    plt.legend(fontsize=20)
    # Choosing B
    plt.subplot(1, 2, 2)
    plt.plot(ratio[:5, 2], "o-", label="B Rewarded", lw=3, ms=10)
    plt.plot(ratio[:5, 3], "o--", label="B Not Rewarded", lw=3, ms=10)
    plt.xlabel("Choice History", fontsize=20)
    plt.ylabel("Likelihood of Choosing B", fontsize=20)
    plt.ylim(0.0, 0.8)
    plt.yticks(fontsize=20)
    plt.xticks(np.arange(5), ["$A^{}$B".format(each) for each in range(1, 6)], fontsize=20)
    plt.legend(fontsize=20)
    plt.show()


def historyAnalysisFor3(choices, rewards):
    choices_sequence = ''.join(np.array(choices.squeeze(), dtype=np.str))
    # Find specific patterns
    repeat_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ratio = np.zeros((len(repeat_list), 6))
    for index, repeat_num in enumerate(repeat_list):
        # print("="*10, repeat_num, "="*10)
        res_iter = _findThreePattern(choices_sequence, repeat_num = repeat_num)
        reward_determin_index = np.array([each[0] for each in res_iter]) + repeat_num
        choice_determin_index = np.array([each[0] for each in res_iter]) + repeat_num + 1
        previous_choice = np.array([each[1] for each in res_iter])
        last_choice = np.array([each[2] for each in res_iter])
        # Determine if the last choice is rewarded or not
        if_rewarded = rewards[reward_determin_index].squeeze()
        if_keep_choice = (choices[choice_determin_index].squeeze() == previous_choice)
        if_choose_another = (choices[choice_determin_index].squeeze() == last_choice)
        if_choose_irrelevant = np.array(
            [
                each != last_choice[index] and each != previous_choice[index]
                for index, each in enumerate(choices[choice_determin_index].squeeze())
            ]
        )
        # Collect data
        reward_index = np.where(if_rewarded == 1)[0]
        not_reward_index = np.where(if_rewarded == 0)[0]
        ratio[index, :] = [
            np.sum(if_keep_choice[reward_index]) / len(reward_index),
            np.sum(if_keep_choice[not_reward_index]) / len(not_reward_index),
            np.sum(if_choose_another[reward_index]) / len(reward_index),
            np.sum(if_choose_another[not_reward_index]) / len(not_reward_index),
            np.sum(if_choose_irrelevant[reward_index]) / len(reward_index),
            np.sum(if_choose_irrelevant[not_reward_index]) / len(not_reward_index)
        ]
    return ratio


def plotAnalysisFor3(ratio, task_name):
    # Plot for reward and not reward
    plt.figure(figsize=(24, 12))
    plt.title(task_name, fontsize = 20)
    # Choosing A
    plt.subplot(1, 3, 1)
    plt.plot(ratio[:5, 0], "o-", label="B Rewarded", lw=3, ms=10)
    plt.plot(ratio[:5, 1], "o--", label="B Not Rewarded", lw=3, ms=10)
    plt.xlabel("Choice History", fontsize=20)
    plt.ylabel("Likelihood of Choosing A", fontsize=20)
    plt.ylim(-0.1, 1.1)
    plt.yticks(fontsize=20)
    plt.xticks(np.arange(5), ["$A^{}$B".format(each) for each in range(1, 6)], fontsize=20)
    plt.legend(fontsize=20)
    # Choosing B
    plt.subplot(1, 3, 2)
    plt.plot(ratio[:5, 2], "o-", label="B Rewarded", lw=3, ms=10)
    plt.plot(ratio[:5, 3], "o--", label="B Not Rewarded", lw=3, ms=10)
    plt.xlabel("Choice History", fontsize=20)
    plt.ylabel("Likelihood of Choosing B", fontsize=20)
    plt.ylim(-0.1, 1.1)
    plt.yticks(fontsize=20)
    plt.xticks(np.arange(5), ["$A^{}$B".format(each) for each in range(1, 6)], fontsize=20)
    plt.legend(fontsize=20)
    # Choosing B
    plt.subplot(1, 3, 3)
    plt.plot(ratio[:5, 4], "o-", label="B Rewarded", lw=3, ms=10)
    plt.plot(ratio[:5, 5], "o--", label="B Not Rewarded", lw=3, ms=10)
    # plt.plot(range(5), [0] * 5, "k-", lw = 2, alpha = 0.5)
    plt.xlabel("Choice History", fontsize=20)
    plt.ylabel("Likelihood of Choosing C", fontsize=20)
    plt.ylim(-0.1, 1.1)
    plt.yticks(fontsize=20)
    plt.xticks(np.arange(5), ["$A^{}$B".format(each) for each in range(1, 6)], fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("three-armed-bandit-task-new-training.pdf")
    plt.show()


def _findTwoPattern(choices, repeat_num = 2):
    # Find pattern for two stimulus
    res_iter = [[m.start(), 1] for m in re.finditer(re.compile("{}2".format("1"*repeat_num)), choices)]
    res_iter.extend([[m.start(), 2] for m in re.finditer(re.compile("{}1".format("2"*repeat_num)), choices)])
    return res_iter


def _findThreePattern(choices, repeat_num = 2):
    # Find pattern for three stimulus
    res_iter = [[m.start(), 1, 2] for m in re.finditer(re.compile("{}2".format("1" * repeat_num)), choices)] # e.g., AAB
    res_iter.extend([[m.start(), 1, 3] for m in re.finditer(re.compile("{}3".format("1" * repeat_num)), choices)]) # e.g., AAC
    res_iter.extend([[m.start(), 2, 1] for m in re.finditer(re.compile("{}1".format("2" * repeat_num)), choices)]) # e.g., BBA
    res_iter.extend([[m.start(), 2, 3] for m in re.finditer(re.compile("{}3".format("2" * repeat_num)), choices)]) # e.g., BBC
    res_iter.extend([[m.start(), 3, 1] for m in re.finditer(re.compile("{}1".format("3" * repeat_num)), choices)])  # e.g., CCA
    res_iter.extend([[m.start(), 3, 2] for m in re.finditer(re.compile("{}2".format("3" * repeat_num)), choices)])  # e.g., CCB
    return res_iter

if __name__ == '__main__':
    # Pre-processing
    config = "ThreeArmed-New"
    # Pre-processing
    if config == "ThreeArmed-Old":
        path = "../RewardAffectData-OldTraining-OldNetwork-Three-Armed-Bandit/"
        log_file_name = path + "RewardAffectData-OldTraining-OldNetwork-ThreeArmed-sudden-reverse-model1-validation-1e6.hdf5"
        data_file_name = path + "data/RewardAffect_ThreeArmed_TestingSet-2020_05_01-blk70-reverseblk0-noise-1.mat"
    elif config == "TwoArmed":
        path = "../Two-Armed-Bandit-SlowReverse/"
        log_file_name = path + "SimplifyTwoArmedSlowReverseNoNoise-validation-15e6.hdf5"
        data_file_name = path + "data/SimplifyTwoArmedSlowReverseWithNoise_TestingSet-2020_04_07-1.mat"
    elif config == "ThreeArmed-New":
        path = "../RewardAffectData-NewTraining-OldNetwork-Three-Armed-Bandit/"
        log_file_name = path + "RewardAffectData-NewTraining-OldNetwork-Three-Armed-slow-reverse-model2-validation-1e6.hdf5"
        data_file_name = path + "data/RewardAffect_ThreeArmed_TestingSet-2020_05_03-blk70-reverseblk5-noise-1.mat"
    else:
        raise ValueError("Undefined task name!")
    choices, rewards = getChoiceAndReward(log_file_name, data_file_name)

    # # For task with two stimulus
    # ratio = historyAnalysisFor2(choices, rewards)
    # plotAnalysisFor2(ratio, "Two-Armed-Bandit")

    # For task with three stimulus
    ratio = historyAnalysisFor3(choices, rewards)
    plotAnalysisFor3(ratio, "Three-Armed-Bandit")