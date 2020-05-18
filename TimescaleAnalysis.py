import h5py
import numpy as np
import statsmodels.api as sm
import pandas as pd
import patsy
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import ExponentialSmoothing


def getFiringRate(logFileName):
    # Read data from HDF5 file
    logFile = h5py.File(logFileName, 'r')
    print(list(logFile.keys()))
    raw_firing_rate = logFile['neuron'].value.squeeze()
    trial_time_index = logFile['index'].value.squeeze()
    choices = logFile['choice'].value
    rewards = logFile['reward'].value
    neuron_num = raw_firing_rate.shape[1]

    # Find unfinished trial
    trial_length = 10
    finished_trial_index = np.array(
        [True if trial_length == (each[1] - each[0]) else False for each in trial_time_index])
    print("Finished trial rate: ", np.sum(finished_trial_index) / len(finished_trial_index))

    # The number of whole blocks
    blk_size = 70
    numTrials = len(finished_trial_index)
    numBlks = numTrials // blk_size if (numTrials // blk_size) % 2 == 0 else (numTrials // blk_size) - 1  # number of small blocks
    numTrials = numBlks * blk_size
    numBlks = numBlks // 2  # The number of whole blocks (before and after reversing)
    firing_rate = raw_firing_rate[:numTrials, :]
    finished_trial_index = finished_trial_index[:numTrials]
    choices = choices[:numTrials].reshape((numBlks, -1))
    rewards = rewards[:numTrials].reshape((numBlks, -1))
    # Pre-compute the finished trial index for every position in a block
    all_index_mat_format = np.arange(0, numTrials).reshape((numBlks, -1))  # reshape all the index for later use
    finished_trial_mat_form = []  # should be a list with length of 2*blk_size; each element is a list containing the index of finished trial for that position
    for index in range(2 * blk_size):
        temp = []
        for each in all_index_mat_format[:, index]:
            if finished_trial_index[each] == True:
                temp.append(each)
        finished_trial_mat_form.append(temp)

    # Reconstruct the firing rate matrix
    # with shape of (number of whole blocks, number of trials in a whole block, time steps of a trial, number of neurons)
    firing_rate_data = np.zeros((numBlks, 2 * blk_size, trial_length, neuron_num))
    for index in range(numTrials):
        blk_index = index // (2 * blk_size)
        trial_index = index % (2 * blk_size)
        # If the trial is finished, take the corresponding hidden unit value as neuron firing rate
        if finished_trial_index[index]:
            firing_rate_data[blk_index, trial_index, :, :] = \
                raw_firing_rate[int(trial_time_index[index][0]):int(trial_time_index[index][1]), :]
        # else, randomly choose a trial from another block at the same position
        else:
            random_finished_trial_index = np.random.choice(finished_trial_mat_form[trial_index], 1).item()
            firing_rate_data[blk_index, trial_index, :, :] = \
                raw_firing_rate[int(trial_time_index[random_finished_trial_index][0]):int(trial_time_index[random_finished_trial_index][1]), :]
    logFile.close()
    return firing_rate_data, choices, rewards

def intrinsicAnalysis(data):
    # TODO: the format of data
    plt.plot(data)
    plt.show()
    model = AutoReg(data, lags = 5).fit()
    return model

def choiceMemoryAnalysis(data):
    # TODO: the format of data
    data = np.mean(data, axis = 1)
    res = ExponentialSmoothing(data).fit(smoothing_level = 0.9)
    plt.plot(data, "bo-", lw=3)
    # plt.plot(res.fittedvalues, "go--", lw = 3)
    plt.plot(res.fittedfcast[1:], "go--", lw = 3)
    plt.ylim(0.5, 1.1)
    plt.show()
    return res




if __name__ == '__main__':
    path = "RewardAffectData-OldTraining-OldNetwork-Three-Armed-Bandit/"
    log_file_name = path + "RewardAffectData-OldTraining-OldNetwork-ThreeArmed-sudden-reverse-model1-validation-1e6.hdf5"
    all_firing_rate, choices, rewards =getFiringRate(log_file_name)
    print("Finished preprocessing!")

    # # Intrinsic timescale analysis
    # res = intrinsicAnalysis(all_firing_rate[:, 20, 2, 1])
    # res.plot_diagnostics()
    # plt.show()
    # print(res.summary())

    # Choice memory timescale analysis
    res = choiceMemoryAnalysis(all_firing_rate[10, :, 0:2, 1]) # 5:7 is the time steps when making a choice

    print(res.summary())
