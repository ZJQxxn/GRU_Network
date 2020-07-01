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



def negativeLogLikelihood(param, choices, rewards):
    '''
    Compute the negative log likelihood of data.
    :param param: Parameters (alpha, beta, gamma, omega)
    :param choices: Choices of validation.
    :param rewards: Rewards of validation.
    :return: Negative log  likelihood.
    '''
    #TODO: check correctness; log 0 smoothing
    alpha = param[0]
    beta = param[1]
    gamma = param[2]
    omega = param[3]
    choices_num = 3 # TODO: for three-armed-bandit task
    trials_num = len(rewards)

    reward_value = np.zeros((choices_num, ))
    choice_value = np.zeros((choices_num, ))
    overall_value = np.zeros((choices_num, ))
    value_trajectory = []
    prob_trajectory = []

    nll = 0 # negative log likelihood
    # param_val = param.valuesdict()
    for trial in range(1, trials_num):
        choice = int(choices[trial] - 1)
        reward = int(rewards[trial])
        # reward-dependent value Q_{t+1}(x) = (1-alpha) * Q_{t}(x) + alpha * Rew_t
        reward_value[choice] = (1 - alpha) * reward_value[choice] + alpha * reward
        # choice-dependent value C_{t+1}(x) = (1-beta) * C_{t}(x) + beta * Cho_t
        choice_value[choice] = (1 - beta) * choice_value[choice] + beta * choice
        # overall value V_{t}(x) = gamma * Q_{t}(x) + (1-gamma) * C_{t}(x)
        overall_value[choice] = gamma * reward_value[choice] + (1-gamma) * choice_value[choice]
        value_trajectory.append(overall_value)
        # negative log likelihood
        weighted_overall_value = omega * overall_value
        exp_overall_value = np.sum(np.exp(weighted_overall_value))
        log_prob = weighted_overall_value[choice] - np.sum(exp_overall_value)
        nll += (- log_prob)
        prob_trajectory.append(exp_overall_value / np.sum(exp_overall_value))
    return nll


def estimationError(param, choices, rewards):
    #TODO: use estimation loss for estimation
    pass


def MLE(logFilename, dataFilename):
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
    # params = lmfit.Parameters()
    # params.add("alpha", value = 0.5, min = 0.0, max = 1.0)
    # params.add("beta", value = 0.5, min = 0.0, max = 1.0)
    # params.add("gamma", value = 0.5, min = 0.0, max = 1.0)
    # params.add("omega", value = 1.0)
    # Minimize negative log likelihood
    bounds = [[0, 1], [0, 1], [0, 1], [None, None]]
    params = np.array([0.5, 0.5, 0.5, 0.5])
    cons = []
    for par in range(len(bounds)):
        if bounds[par][0] is not None:
            l = {"type": "ineq", "fun": lambda x: x[par] - bounds[par][0]}
            cons.append(l)
        if bounds[par][1] is not None:
            u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
            cons.append(u)
    func = lambda parameter: negativeLogLikelihood(parameter, choices, rewards)
    res = scipy.optimize.minimize(
        func,
        x0 = params,
        method = "SLSQP",
        bounds = bounds,
        # constraints = cons
    )
    # res = lmfit.minimize(negativeLogLikelihood, params, args = choices, kws = {"rewards": rewards})
    print(res)
    print("Estimated Parameter : ", res.x)



if __name__ == '__main__':
    # Configurations
    path = "RewardAffectData-OldTraining-OldNetwork-Three-Armed-Bandit/"
    validation_log_filename = path + "RewardAffectData-OldTraining-OldNetwork-ThreeArmed-sudden-reverse-model1-validation-1e6.hdf5"
    testing_data_filename = path + "data/RewardAffect_ThreeArmed_TestingSet-2020_05_01-blk70-reverseblk0-noise-1.mat"
    # MLE for parameter estimation
    MLE(validation_log_filename, testing_data_filename)