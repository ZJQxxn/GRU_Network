'''
Description:
    Behavioral timescale analysis through inverse reinforcement learning.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date: 
    Jun. 30 2020
'''

import pandas as pd
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat
import scipy.optimize
import scipy.stats
import csv
import pickle



def negativeLogLikelihood(param, choices, rewards, return_trajectory = False):
    '''
    Compute the negative log likelihood of data.
    :param param: Parameters (alpha, beta, gamma, omega)
    :param choices: Choices of validation.
    :param rewards: Rewards of validation.
    :param return_trajectory: Boolean value determine whether return the trajectory of probability  and value function.
    :return: If return_trajectory is true: return negative log likelihood, value function trajectory, probability trajectory.
    '''
    alpha = param[0]
    beta = param[1]
    gamma = param[2]
    omega = param[3]
    choices_num = 3 # TODO: for three-armed-bandit task; generalize later
    trials_num = len(rewards)

    reward_value = np.zeros((choices_num, ))
    choice_value = np.zeros((choices_num, ))
    overall_value = np.zeros((choices_num, ))
    value_trajectory = []
    prob_trajectory = []

    nll = 0 # negative log likelihood
    # param_val = param.valuesdict()
    for trial in range(trials_num):
        choice = int(choices[trial] - 1)
        reward = int(rewards[trial])
        # reward-dependent value Q_{t+1}(x) = (1-alpha) * Q_{t}(x) + alpha * Rew_t
        reward_value[choice] = (1 - alpha) * reward_value[choice] + alpha * reward
        # choice-dependent value C_{t+1}(x) = (1-beta) * C_{t}(x) + beta * Cho_t
        choice_value[choice] = (1 - beta) * choice_value[choice] + beta * (choice + 1)
        # overall value V_{t}(x) = gamma * Q_{t}(x) + (1-gamma) * C_{t}(x)
        overall_value[choice] = gamma * reward_value[choice] + (1-gamma) * choice_value[choice]
        value_trajectory.append(overall_value)
        # negative log likelihood
        weighted_overall_value = omega * overall_value
        exp_overall_value = np.exp(weighted_overall_value) + 1e-5
        log_prob = weighted_overall_value[choice] - np.log(np.sum(exp_overall_value))
        nll += (- log_prob)
        prob_trajectory.append(exp_overall_value / np.sum(exp_overall_value))
    if not return_trajectory:
        return nll
    else:
         return (nll, value_trajectory, prob_trajectory)


def estimationError(param, choices, rewards, true_reward_prob, return_trajectory = False):
    '''
    Compute the negative log likelihood of data.
    :param param: Parameters (alpha, beta, gamma, omega)
    :param choices: Choices of validation.
    :param rewards: Rewards of validation.
    :param true_reward_prob: True reward probability for each trial.
    :param return_trajectory: Boolean value determine whether return the trajectory of probability  and value function.
    :return: If return_trajectory is true: return negative log likelihood, value function trajectory, probability trajectory.
    '''
    alpha = param[0]
    beta = param[1]
    gamma = param[2]
    omega = param[3]
    choices_num = 3  # TODO: for three-armed-bandit task; generalize later
    trials_num = len(rewards)

    reward_value = np.zeros((choices_num,))
    choice_value = np.zeros((choices_num,))
    overall_value = np.zeros((choices_num,))
    value_trajectory = []
    prob_trajectory = []

    estimation_error = 0  # negative log likelihood
    # param_val = param.valuesdict()
    for trial in range(trials_num):
        choice = int(choices[trial] - 1)
        reward = int(rewards[trial])
        # reward-dependent value Q_{t+1}(x) = (1-alpha) * Q_{t}(x) + alpha * Rew_t
        reward_value[choice] = (1 - alpha) * reward_value[choice] + alpha * reward
        # choice-dependent value C_{t+1}(x) = (1-beta) * C_{t}(x) + beta * Cho_t
        choice_value[choice] = (1 - beta) * choice_value[choice] + beta * (choice + 1)
        # overall value V_{t}(x) = gamma * Q_{t}(x) + (1-gamma) * C_{t}(x)
        overall_value[choice] = gamma * reward_value[choice] + (1 - gamma) * choice_value[choice]
        value_trajectory.append(overall_value)
        # negative log likelihood
        weighted_overall_value = omega * overall_value
        exp_overall_value = np.exp(weighted_overall_value)
        estimation_prob = exp_overall_value / np.sum(exp_overall_value)
        estimation_error += np.linalg.norm(estimation_prob - true_reward_prob[:, trial])**2
        prob_trajectory.append(exp_overall_value / np.sum(exp_overall_value))
    if not return_trajectory:
        return estimation_error
    else:
        return (estimation_error, value_trajectory, prob_trajectory)


def MLE(logFilename, dataFilename, block_size = 70):
    '''
    Maximize likelihood estimation for learning the paramteters.
    :param logFilename: Filename of the validation log.
    :param dataFilename: Filename of the validation dataset.
    :return: VOID
    '''
    # Load data; Pre-computing
    logFile = h5py.File(logFilename, 'r')
    choices = logFile['choice'].value.squeeze()
    rewards = logFile['reward'].value.squeeze()
    num_trials = len(choices)
    reward_prob = loadmat(dataFilename)['data_ST_Brief'][0][0][0]
    for index in range(1, num_trials):
        if choices[index] > 3: # unfinished trial
            choices[index] = choices[index - 1]
            rewards[index] = int(np.random.uniform(0, 1, 1) < reward_prob[int(choices[index] - 1)][index])
    print("Number of trials (samples): %d" % num_trials)
    # Create parameters with constraints
    bounds = [[0, 1], [0, 1], [0, 1], [None, None]]
    params = np.array([0.5, 0.5, 0.5, 1])
    func = lambda parameter: negativeLogLikelihood(parameter, choices, rewards)
    success = False
    if not success:
        res = scipy.optimize.minimize(
            func,
            x0 = params,
            method = "SLSQP",
            bounds = bounds,
            tol = 1e-8 # improve convergence
        )
        success = res.success
        if not success:
            print("Fail. Retry...")
    # print("Success : ", res.success)
    print("Estimated Parameter (alpha, beta, gamma, omega): ", res.x)
    # Test the estimation
    _, value_function, estimation_prob = negativeLogLikelihood(res.x, choices, rewards, return_trajectory = True)
    estimation_choices = np.array([np.argmax(each) for each in estimation_prob]) + 1
    correct_rate = np.sum(estimation_choices == choices) / len(choices)
    print("Estimation correct rate : ", correct_rate)
    # Write to file
    np.save("Behavioral-Analysis-Result/MLE_choice_estimation.npy", estimation_prob)
    np.save("Behavioral-Analysis-Result/MLE_parameter_estimation.npy", res.x)
    # Plot the estimated choices
    analysis(estimation_choices, reward_prob, "MLE", block_size)


def MEE(logFilename, dataFilename, block_size = 70):
    '''
    Minimize the square error for learrning the parameters.
    :param logFilename: Filename of the validation log.
    :param dataFilename: Filename of the validation dataset.
    :return: VOID
    '''
    # Load data; Pre-computing
    logFile = h5py.File(logFilename, 'r')
    choices = logFile['choice'].value.squeeze()
    rewards = logFile['reward'].value.squeeze()
    num_trials = len(choices)
    reward_prob = loadmat(dataFilename)['data_ST_Brief'][0][0][0]
    for index in range(1, num_trials):
        if choices[index] > 3:  # unfinished trial
            choices[index] = choices[index - 1]
            rewards[index] = int(np.random.uniform(0, 1, 1) < reward_prob[int(choices[index] - 1)][index])
    print("Number of trials (samples): %d" % num_trials)
    # Create parameters with constraints
    bounds = [[0, 1], [0, 1], [0, 1], [None, None]]
    params = np.array([0.5, 0.5, 0.5, 1])
    func = lambda parameter: estimationError(parameter, choices, rewards, reward_prob)
    success = False
    if not success:
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds,
            tol=1e-8  # improve convergence
        )
        success = res.success
        if not success:
            print("Fail. Retry...")
    # print("Success : ", res.success)
    print("Estimated Parameter (alpha, beta, gamma, omega): ", res.x)
    # Test the estimation
    _, value_function, estimation_prob = estimationError(res.x, choices, rewards, reward_prob, return_trajectory=True)
    estimation_choices = np.array([np.argmax(each) for each in estimation_prob]) + 1
    correct_rate = np.sum(estimation_choices == choices) / len(choices)
    print("Estimation correct rate : ", correct_rate)
    # Write to file
    np.save("Behavioral-Analysis-Result/MEE_choice_estimation.npy", estimation_prob)
    np.save("Behavioral-Analysis-Result/MEE_parameter_estimation.npy", res.x)
    # Plot the estimated choices
    analysis(estimation_choices, reward_prob, "MEE", block_size, )


def analysis(choices, reward_probability, method, block_size = 70):
    # Compute objective highest reward probability
    reward_probability = reward_probability[:,:2*block_size]
    objective_highest = []
    trial_num = reward_probability.shape[1]
    for i in range(trial_num):
        trial_reward = reward_probability[:, i]
        max_ind = np.argwhere(trial_reward == np.amax(trial_reward))
        max_ind = max_ind.flatten()
        if 1 == len(max_ind.shape):  # only one stimulus has the highest reward probability
            objective_highest.append([max_ind[0], trial_reward[max_ind[0]]])
        elif 2 == len(max_ind.shape):  # two stimulus has the highest reward probability
            # 3 for A/B, 4 for A/C, 5 for B/C
            highest_reward = trial_reward[0] if 0 in max_ind else trial_reward[1]
            objective_highest.append([np.sum(max_ind) + 2, trial_reward[0], highest_reward])
        else:  # all the stimulus has the same reward probability
            objective_highest.append([6, trial_reward[0]])
    objective_highest = np.array(objective_highest)
    # Compute experienced reward probability
    trial_num = choices.shape[0]
    block_num = trial_num // (block_size * 2)
    block_reward_prob = reward_probability[:, :2*block_size]
    experienced_reward_prob = np.zeros((block_size * 2, block_num)) # Experienced reward probability as a (number of trials in one block, number of blocks) matrix
    for index, choice in enumerate(choices):
        index_in_block = index % (2 * block_size)
        block_index = index // (2 * block_size)
        if block_index >= block_num:
            break
        if choice.item() > 3:
            experienced_reward_prob[index_in_block, block_index] = np.nan
        else:
            experienced_reward_prob[index_in_block, block_index] = block_reward_prob[choice.item() - 1, index_in_block]
    # Compute random reward probability
    random_reward = np.array(
        [block_reward_prob[np.random.choice([0, 1, 2], 1), index % block_reward_prob.shape[1]] for index in
         range(experienced_reward_prob.shape[0] * experienced_reward_prob.shape[1])]) \
        .reshape(experienced_reward_prob.shape)
    # ================== PLOT PROBABILITY ==================
    mean_experienced_reward_prob = np.nanmean(experienced_reward_prob, axis=1)  # average value
    SEM_experienced_reward_prob = scipy.stats.sem(experienced_reward_prob, axis=1)  # SEM
    # Show H_sch
    for i in range(2 * block_size):
        temp = objective_highest[i, :]
        if temp[0] == 0:
            color = 'red'
        elif temp[0] == 1:
            color = 'blue'
        elif temp[0] == 2:
            color = 'green'
        else:
            color = 'cyan'
        plt.scatter(i, temp[1], color=color)
    plt.title("Experienced Reward Prob. vs. Random Reward Prob.", fontsize=20)
    plt.plot(np.arange(0, block_size * 2), objective_highest[:, 1], 'k-', ms=8)
    # Plot experienced reward probability
    plt.plot(np.arange(0, block_size * 2), mean_experienced_reward_prob, 's-m',
             label="Experienced Reward Prob ({})".format(method), ms=8, lw=2)
    plt.fill_between(np.arange(0, block_size * 2),
                     mean_experienced_reward_prob - SEM_experienced_reward_prob,
                     mean_experienced_reward_prob + SEM_experienced_reward_prob,
                     color = "#dcb2ed",
                     alpha = 0.8,
                     linewidth = 4)
    plt.plot(np.mean(random_reward, axis=1), 'b--', alpha=0.5,
             label="Random Reward Prob.", lw=2)
    plt.ylim((0.0, 0.85))
    plt.yticks(fontsize=20)
    plt.ylabel("Reward Probability", fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel("Trial", fontsize=20)
    plt.legend(loc="best", fontsize=20)
    plt.show()


def correlation():
    pass


if __name__ == '__main__':
    # Configurations
    path = "RewardAffectData-OldTraining-OldNetwork-Three-Armed-Bandit/"
    validation_log_filename = path + "RewardAffectData-OldTraining-OldNetwork-ThreeArmed-sudden-reverse-model1-validation-1e6.hdf5"
    testing_data_filename = path + "data/RewardAffect_ThreeArmed_TestingSet-2020_05_01-blk70-reverseblk0-noise-1.mat"

    # # MLE for parameter estimation
    print("="*10, " MLE ", "="*10)
    MLE(validation_log_filename, testing_data_filename, block_size = 70)

    # MSE for parameter estimation
    print("="*10, " MSE ", "="*10)
    MEE(validation_log_filename, testing_data_filename, block_size = 70)