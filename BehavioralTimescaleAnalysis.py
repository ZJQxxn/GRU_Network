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
        exp_overall_value = np.exp(weighted_overall_value)
        log_prob = weighted_overall_value[choice] - np.sum(exp_overall_value)
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


def MLE(logFilename, dataFilename):
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
    #TODO: use the estimation for the same analysis as three-armed-task


def MSE(logFilename, dataFilename):
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
    # TODO: use the estimation for the same analysis as three-armed-task


if __name__ == '__main__':
    # Configurations
    path = "RewardAffectData-OldTraining-OldNetwork-Three-Armed-Bandit/"
    validation_log_filename = path + "RewardAffectData-OldTraining-OldNetwork-ThreeArmed-sudden-reverse-model1-validation-1e6.hdf5"
    testing_data_filename = path + "data/RewardAffect_ThreeArmed_TestingSet-2020_05_01-blk70-reverseblk0-noise-1.mat"

    # # MLE for parameter estimation
    print("="*10, " MLE ", "="*10)
    MLE(validation_log_filename, testing_data_filename)

    # MSE for parameter estimation
    print("="*10, " MSE ", "="*10)
    MSE(validation_log_filename, testing_data_filename)