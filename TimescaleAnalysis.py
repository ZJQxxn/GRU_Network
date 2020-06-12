import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot_2samples
from scipy.stats import sem
from matplotlib_venn import venn3
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# ==================================================================
#                          PRE-PROCESSING
# ==================================================================

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


# ==================================================================
#                TIMESCALE ANALYSIS WITH AUTO-REGRESSIVE
# ==================================================================

def intrinsicAnalysis(data, lags):
    '''
    Intrinsic timescale analysis.
    :param data: Firing data with the shape of (block num, trial num in a block, time steps, neuron num) 
    :param lags: Time step lags for auto-regressive.
    :return: Fitted auto-regressive models for every neuron. The auto-regressive coefficients.
        all_models: A list with each element a fittted auto-regressive model (statsmodels.tsa.ar_model.AutoReg).
        autoreg_res: A matrix containing auto-regressive coefficient of all the neurons with the shape of (neuron num, lags). 
    '''
    neurons_num = data.shape[-1]
    mat_data = np.zeros((np.prod(data.shape[:-1]), neurons_num))
    # data = data.reshape((-1, neurons_num))
    for index in range(neurons_num):
        mat_data[:, index] = data[:,:,:,index].reshape(-1)
    print("Data shape:", mat_data.shape)
    # ====================================
    #           An Example
    # ====================================
    # plot part of samples
    plt.plot(mat_data[:500, 0])
    plt.show()
    # plot auto-correlation of samples
    plot_acf(mat_data[:, 0])
    plt.show()
    # plot partial auto-correlation of samples
    plot_pacf(mat_data[:, 0])
    plt.show()
    # Auto-regressive
    autoreg_res = np.zeros((neurons_num, lags)) # the auto-regressive coefficients for each neuron
    all_models = []
    for index in range(neurons_num):
        neuron_data = mat_data[:, index]
        # filter out useless neurons
        if np.all(0 == neuron_data):
            continue
        model = AutoReg(neuron_data, lags = lags).fit()
        autoreg_res[index, :] = np.abs(model.params[1:])
        all_models.append(model)
    # Filter out useless neurons
    for index in range(neurons_num):
        temp = autoreg_res[index, :]
        if np.all(0 == temp):
            autoreg_res[index, :] = np.tile(np.nan, len(temp))
    return all_models, autoreg_res


def plotIntrinsicResult(all_models, autoreg_res, lags):
    '''
    Plot intrisic timescale analysis results.
    :param all_models: All the auto-regressive models for every node.
    :param autoreg_res: Auto-regressive coefficient matrix.
    :return: VOID
    '''
    # Plot mean and SEM of auto-regressive coefficients
    coeff_sem = sem(autoreg_res, axis=0, nan_policy = 'omit')
    coeff_mean = np.nanmean(autoreg_res, axis=0)
    # # Normalization
    # coeff_mean = coeff_mean / np.sum(coeff_mean)
    # Plot the AR coefficient
    plt.title("AR Coef. vs. Time Lags [Intrinsic]", fontsize=20)
    plt.plot(coeff_mean, ms=8, lw=2)
    plt.fill_between(np.arange(0, lags),
                     coeff_mean - coeff_sem,
                     coeff_mean + coeff_sem,
                     color="#dcb2ed",
                     alpha=0.5,
                     linewidth=4)
    # plt.plot(autoreg_res[2,:], "bo-", ms=8, lw=2)
    plt.xticks([0, 4, 9, 14, 19, 24, 29, 34, 39, 44], [1, 5, 10, 15, 20, 25, 30, 35, 40, 45])# AR(45)
    plt.xlabel("Time Lag", fontsize=20)
    plt.ylabel("AR Coef.", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.0, 0.6)
    plt.show()


def trialLevelSeasonalAnalysis(data, lags):
    '''
        Trial-level seasonal timescale analysis.
        :param data: Firing data with the shape of (block num, trial num in a block, time steps, neuron num) 
        :param lags: Time step lags for auto-regressive.
        :return: Fitted auto-regressive models for every neuron. The auto-regressive coefficients.
            all_models: A list with each element a fittted auto-regressive model (statsmodels.tsa.ar_model.AutoReg).
            autoreg_res: A matrix containing auto-regressive coefficient of all the neurons with the shape of (neuron num, lags). 
        '''
    data_shape = data.shape
    neurons_num = data_shape[-1]
    time_step_num = data_shape[2]
    firing_data = np.zeros(
        (data_shape[0] * data_shape[1], time_step_num, neurons_num)
    )
    for i in range(neurons_num):
        for j in range(time_step_num):
            firing_data[:, j, i] = data[:, :, j, i].reshape(-1)
    print("Data shape:", firing_data.shape)
    # ====================================
    #           An Example
    # ====================================
    # plot part of samples
    plt.plot(firing_data[:200, 5, 0])
    plt.show()
    # plot auto-correlation of samples
    plot_acf(firing_data[:, 5, 0])
    plt.show()
    # plot partial auto-correlation of samples
    plot_pacf(firing_data[:, 5, 0])
    plt.show()
    # Auto-regressive
    autoreg_res = np.zeros((neurons_num, time_step_num, lags))  # the auto-regressive coefficients for each neuron and time step
    all_models = []
    for i in range(neurons_num):
        model_per_time_step = []
        for j in range(time_step_num):
            neuron_data = firing_data[:, j, i]
            # filter out useless neurons
            if np.all(0 == neuron_data):
                continue
            model = AutoReg(neuron_data, lags=lags).fit()
            autoreg_res[i, j, :] = np.abs(model.params[1:])
            model_per_time_step.append(model)
        all_models.append(model_per_time_step)
    #TODO: 数据不平均，因为对于每个 time step， 筛选出的 neuron 数量不同。存在很多 0 值
    return all_models, autoreg_res


def plotTrialSeasonalResult(all_models, autoreg_res, lags):
    '''
    Plot trial-level seasonal timescale analysis results.
    :param all_models: All the auto-regressive models for every node.
    :param autoreg_res: Auto-regressive coefficient matrix.
    :return: VOID
    '''
    # Plot mean and SEM of auto-regressive coefficients
    coeff_sem = sem(autoreg_res, axis=0)
    coeff_mean = np.mean(autoreg_res, axis = 0)
    # Plot the AR coefficient
    plt.title("AR Coef. vs. Time Lags [Trial-Level Seasonal]", fontsize=20)
    # for index in range(coeff_mean.shape[0]):
    plt.plot(np.mean(coeff_mean[[0,1,9]], axis = 0), "o-", ms=8, lw=2, label = "reset trial")
    plt.plot(np.mean(coeff_mean[[2,3,4]], axis = 0), "o-", ms=8, lw=2, label = "show stimulus")
    plt.plot(np.mean(coeff_mean[[5,6]], axis = 0), "o-", ms=8, lw=2, label = "make choice")
    plt.plot(np.mean(coeff_mean[[7,8]], axis = 0), "o-", ms=8, lw=2, label = "show reward")
    # plt.fill_between(np.arange(0, lags),
    #                 coeff_mean[[0,1,9]] - coeff_sem[[0,1,9]],
    #                 coeff_mean[[0,1,9]] + coeff_sem[[0,1,9]],
    #                 color="#dcb2ed",
    #                 alpha=0.5,
    #                 linewidth=4,
    #                 label = "SEM"
    #                 )
    plt.xticks(np.arange(len(coeff_mean[0])), np.arange(1, len(coeff_mean[0]) + 1))
    plt.xlabel("Time Lag", fontsize=20)
    plt.ylabel("AR Coef.", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.1, 0.3)
    plt.legend(fontsize = 20)
    plt.show()


def blockLevelSeasonalAnalysis(data, lags):
    '''
    Block-level seasonal timescale analysis.
    :param data: Firing data with the shape of (block num, trial num in a block, time steps, neuron num) 
    :param lags: Time step lags for auto-regressive.
    :return: Fitted auto-regressive models for every neuron. The auto-regressive coefficients.
        all_models: A list with each element a fittted auto-regressive model (statsmodels.tsa.ar_model.AutoReg).
        autoreg_res: A matrix containing auto-regressive coefficient of all the neurons with the shape of (neuron num, lags). 
    '''
    data_shape = data.shape
    neurons_num = data_shape[-1]
    trial_num = data_shape[1]
    firing_data = np.zeros(
        (data_shape[0] * trial_num, neurons_num)
    )
    for i in range(neurons_num):
        trial_mean_firing_rate = np.mean(data[:, :, :, i], axis = 2)
        firing_data[:, i] = trial_mean_firing_rate.reshape(-1)
    print("Data shape:", firing_data.shape)
    # ====================================
    #           An Example
    # ====================================
    # plot part of samples
    plt.plot(firing_data[:, 0])
    plt.show()
    # plot auto-correlation of samples
    plot_acf(firing_data[:, 0])
    plt.show()
    # plot partial auto-correlation of samples
    plot_pacf(firing_data[:, 0])
    plt.show()
    # Auto-regressive
    autoreg_res = np.zeros(
        (neurons_num, lags))  # the auto-regressive coefficients for each neuron and time step
    all_models = []
    for i in range(neurons_num):
        neuron_data = firing_data[:, i]
        # filter out useless neurons
        if np.all(0 == neuron_data):
            continue
        model = AutoReg(neuron_data, lags=lags).fit()
        autoreg_res[i, :] = np.abs(model.params[1:])
        all_models.append(model)
    # TODO: 数据不平均，因为对于每个 time step， 筛选出的 neuron 数量不同。存在很多 0 值
    return all_models, autoreg_res


def plotBlockSeasonalResult(all_models, autoreg_res, lags):
    '''
    Plot block-level seasonal timescale analysis results.
    :param all_models: All the auto-regressive models for every node.
    :param autoreg_res: Auto-regressive coefficient matrix.
    :return: VOID
    '''
    # Plot mean and SEM of auto-regressive coefficients
    coeff_sem = sem(autoreg_res, axis=0)
    coeff_mean = np.mean(autoreg_res, axis=0)
    # Plot the AR coefficient
    plt.title("AR Coef. vs. Time Lags [Block-Level Seasonal]", fontsize=20)
    plt.plot(coeff_mean, ms=8, lw=2)
    plt.fill_between(np.arange(0, lags),
                     coeff_mean - coeff_sem,
                     coeff_mean + coeff_sem,
                     color="#dcb2ed",
                     alpha=0.5,
                     linewidth=4)
    plt.xticks(np.arange(len(autoreg_res[0])), np.arange(1, len(autoreg_res[0]) + 1))
    plt.xlabel("Time Lag", fontsize=20)
    plt.ylabel("AR Coef.", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.1, 0.3)
    plt.show()


# ==================================================================
#            TIMESCALE ANALYSIS WITH EXPONENTIAL SMOOTHING
# ==================================================================

def choiceMemoryAnalysis(data, choices):
    #TODO: how to use choices
    data_shapes = data.shape
    neurons_num = data_shapes[-1]
    firing_data = data.reshape((-1, neurons_num))
    print("Data shape:", firing_data.shape)
    # ====================================
    #           An Example
    # ====================================
    # plot part of samples
    plt.plot(firing_data[:500, 1])
    plt.show()
    # plot auto-correlation of samples
    plot_acf(firing_data[:, 1])
    plt.show()
    # plot partial auto-correlation of samples
    plot_pacf(firing_data[:, 1])
    plt.show()
    # Exponential smoothing fitted res
    res = ExponentialSmoothing(firing_data[:, 1]).fit()
    plt.plot(firing_data[:,1], "bo-", lw=3)
    # plt.plot(res.fittedvalues, "go--", lw = 3)
    plt.plot(res.fittedfcast[1:], "go--", lw = 3)
    plt.ylim(-0.1, 1.1)
    plt.show()
    return res


def rewardMemoryAnalysis(data, rewards):
    pass



# ==================================================================
#            REWARD & CHOICE ANALYSIS WITH AUTOREGRESSIVE
# ==================================================================

def choiceARAnalysis(choices, lags = 5):
    choices = choices.reshape(-1)
    for index in range(len(choices) - 1):
        if choices[index+1] > 3:
            choices[index+1] = choices[index]
    print("Data shape:", choices.shape)
    # ====================================
    #           An Example
    # ====================================
    # plot part of samples
    sbn.distplot(
        choices,
        bins = [1, 2, 3, 4],
        kde = False,
        hist_kws = {
            "align":"mid",
            "linewidth": 3,
            "alpha": 1
        }
    )
    plt.xticks([1.5, 2.5, 3.5], ["A", "B", "C"], fontsize = 20)
    plt.yticks(fontsize = 12)
    plt.xlabel("Choices", fontsize = 20)
    plt.ylabel("# Trials", fontsize = 20)
    plt.show()
    # plot auto-correlation of samples
    plot_acf(choices)
    plt.show()
    # plot partial auto-correlation of samples
    plot_pacf(choices)
    plt.show()
    # Auto-regressive
    # autoreg_res = np.zeros((neurons_num, lags))  # the auto-regressive coefficients for each neuron
    # all_models = []
    model = AutoReg(choices, lags=lags).fit()
    autoreg_res = np.abs(model.params[1:])
    return model, autoreg_res


def plotChoiceARResult(model, autoreg_res, lags):
    '''
    Plot choice AR analysis results.
    :param all_models: All the auto-regressive models for every node.
    :param autoreg_res: Auto-regressive coefficient matrix.
    :return: VOID
    '''
    # Plot mean and SEM of auto-regressive coefficients
    plt.title("AR Coef. vs. Time Lags [Choice]", fontsize=20)
    plt.plot(autoreg_res, ms=8, lw=2)
    plt.xticks(np.arange(len(autoreg_res)), np.arange(1, len(autoreg_res) + 1))
    plt.xlabel("Time Lag", fontsize=20)
    plt.ylabel("AR Coef.", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.show()


def rewardARAnalysis(rewards, lags = 5):
    rewards = rewards.reshape(-1)
    print("Data shape:", rewards.shape)
    # ====================================
    #           An Example
    # ====================================
    # plot part of samples
    plt.bar([0, 1], [np.sum(rewards == 0), np.sum(rewards == 1)], width = 1, align='edge')
    plt.xticks([0.5, 1.5], ["Not Rewarded", "Rewarded"], fontsize = 20)
    plt.yticks(fontsize = 12)
    # plt.xlabel("R", fontsize = 20)
    plt.ylabel("# Trials", fontsize = 20)
    plt.show()
    # plot auto-correlation of samples
    plot_acf(rewards)
    plt.show()
    # plot partial auto-correlation of samples
    plot_pacf(rewards)
    plt.show()
    # Auto-regressive
    # autoreg_res = np.zeros((neurons_num, lags))  # the auto-regressive coefficients for each neuron
    # all_models = []
    model = AutoReg(rewards, lags=lags).fit()
    autoreg_res = np.abs(model.params[1:])
    return model, autoreg_res


def plotRewardARResult(model, autoreg_res, lags):
    '''
    Plot choice AR analysis results.
    :param all_models: All the auto-regressive models for every node.
    :param autoreg_res: Auto-regressive coefficient matrix.
    :return: VOID
    '''
    # Plot mean and SEM of auto-regressive coefficients
    plt.title("AR Coef. vs. Time Lags [Reward]", fontsize=20)
    plt.plot(autoreg_res, ms=8, lw=2)
    plt.xticks(np.arange(len(autoreg_res)), np.arange(1, len(autoreg_res) + 1))
    plt.xlabel("Time Lag", fontsize=20)
    plt.ylabel("AR Coef.", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.show()



# ==================================================================
#                   ANALYSIS FOR NEURON REGIONS
# ==================================================================

def cateogorizeNeuron(firing_data, time_step_category, need_plot = True):
    '''
    Categorize neurons based on the activation time.
    :param firing_data: Firing rate.
    :return: A dict represent regions of each neuron.
    '''
    data_shape = firing_data.shape
    firing_data = firing_data.reshape(
        (-1, data_shape[2], data_shape[3])
    )
    regions = {}
    for index in range(data_shape[2]):
        time_step_data = firing_data[:, index, :]
        regions[index] = []
        for n in range(data_shape[3]):
            if (np.sum(time_step_data[:, n] != 0) / len(time_step_data[:, n])) > 0.9:
                regions[index].append(n)
    integrate_regions = {}
    for i in range(len(time_step_category)):
        integrate_regions[i] = []
        for step in time_step_category[i]:
            integrate_regions[i].extend(regions[step])
        integrate_regions[i] = list(set(integrate_regions[i]))
    # Plot venn graph
    if need_plot:
        plt.title("Neuron Categories", fontsize = 20)
        venn3(
            [set(integrate_regions[1]), set(integrate_regions[2]), set(integrate_regions[3])],
            set_labels = ("show stimulus", "make choice", "get reward"))
        plt.show()
    return integrate_regions


def regionIntrinsicAnalysis(data, lags, time_step_category):
    '''
    Intrinsic timescale analysis.
    :param data: Firing data with the shape of (block num, trial num in a block, time steps, neuron num) 
    :param lags: Time step lags for auto-regressive.
    :return: Fitted auto-regressive models for every neuron. The auto-regressive coefficients.
        all_models: A list with each element a fittted auto-regressive model (statsmodels.tsa.ar_model.AutoReg).
        autoreg_res: A matrix containing auto-regressive coefficient of all the neurons with the shape of (neuron num, lags). 
    '''
    neurons_num = data.shape[-1]
    mat_data = np.zeros((np.prod(data.shape[:-1]), neurons_num))
    # data = data.reshape((-1, neurons_num))
    for index in range(neurons_num):
        mat_data[:, index] = data[:,:,:,index].reshape(-1)
    print("Data shape:", mat_data.shape)
    # ====================================
    #         Different Region
    # ====================================
    neuron_region = cateogorizeNeuron(data, time_step_category, need_plot = False)
    regions_res = {}
    regions_model = {}
    region_timescale = []
    significant_ratio = []
    for region in neuron_region:
        neurons_index = neuron_region[region]
        neurons_num = len(neurons_index)
        # Auto-regressive
        autoreg_res = np.zeros((neurons_num, lags))  # the auto-regressive coefficients for each neuron
        all_models = []
        region_data = mat_data[:, neurons_index]
        for index in range(neurons_num):
            neuron_data = region_data[:, index]
            # filter out useless neurons
            if np.all(0 == neuron_data):
                continue
            model = AutoReg(neuron_data, lags=lags).fit()
            autoreg_res[index, :] = np.abs(model.params[1:])
            all_models.append(model)
        # Filter out useless neurons
        for index in range(neurons_num):
            temp = autoreg_res[index, :]
            if np.all(0 == temp):
                autoreg_res[index, :] = np.tile(np.nan, len(temp))
        # Find the most significant timescale
        # print(np.min(autoreg_res, axis = 1))
        amplitude = np.arange(lags) + 1
        autoreg_res = -amplitude / np.log(np.abs(autoreg_res))
        # # TODO: take out some significantly large timescale
        # extreme_index = np.where(autoreg_res > 100)
        # autoreg_res[extreme_index] = 0
        # region_timescale.append(np.max(autoreg_res))
        # # TODO: find significant neuron; larger than median
        # median_timescale = np.median(autoreg_res)
        # significant_ratio.append(np.sum(np.max(autoreg_res, axis = 1) > median_timescale) / neurons_num)
        region_timescale.append(np.median(autoreg_res))
        regions_res[region] = autoreg_res
        regions_model[region] = all_models
    # Plot the timescale
    region_timescale = np.array(region_timescale)
    print("Region Timescale: ", region_timescale)
    # print("Region Significant Neurons Ratio: ", significant_ratio)
    sort_index = np.argsort(region_timescale)
    plt.figure(figsize=(7, 10))
    plt.title("Intrinsic Timescale", fontsize = 20)
    plt.plot(region_timescale[sort_index], "bo--", lw = 3, ms = 10)
    plt.xticks(np.arange(len(region_timescale)), np.array(["reset", "show", "choose", "reward"])[sort_index],
               fontsize = 20, rotation = 30)
    plt.yticks(fontsize = 10)
    plt.ylabel("Timescale", fontsize = 20)
    # plt.ylim(0.3, 0.4)
    plt.show()
    return regions_model, regions_res


def regionTrialSeasonalAnalysis(data, lags, time_step_category):
    '''
    Intrinsic timescale analysis.
    :param data: Firing data with the shape of (block num, trial num in a block, time steps, neuron num) 
    :param lags: Time step lags for auto-regressive.
    :return: Fitted auto-regressive models for every neuron. The auto-regressive coefficients.
        all_models: A list with each element a fittted auto-regressive model (statsmodels.tsa.ar_model.AutoReg).
        autoreg_res: A matrix containing auto-regressive coefficient of all the neurons with the shape of (neuron num, lags). 
    '''
    data_shape = data.shape
    neurons_num = data.shape[-1]
    time_step_num = data_shape[2]
    firing_data = np.zeros(
        (data_shape[0] * data_shape[1], time_step_num, neurons_num)
    )
    for i in range(neurons_num):
        for j in range(time_step_num):
            firing_data[:, j, i] = data[:, :, j, i].reshape(-1)
    print("Data shape:", firing_data.shape)
    # ====================================
    #         Different Region
    # ====================================
    neuron_region = cateogorizeNeuron(data, time_step_category, need_plot = False)
    regions_res = {}
    regions_model = {}
    region_timescale = []
    for region in neuron_region:
        neurons_index = neuron_region[region]
        neurons_num = len(neurons_index)
        # Auto-regressive
        autoreg_res = np.zeros((neurons_num, time_step_num, lags))  # the auto-regressive coefficients for each neuron and time step
        all_models = []
        region_data = firing_data[:, :, neurons_index]
        for i in range(neurons_num):
            model_per_time_step = []
            for j in range(time_step_num):
                neuron_data = region_data[:, j, i]
                # filter out useless neurons
                if np.all(0 == neuron_data):
                    continue
                model = AutoReg(neuron_data, lags=lags).fit()
                autoreg_res[i, j, :] = np.abs(model.params[1:])
                model_per_time_step.append(model)
            all_models.append(model_per_time_step)
        autoreg_res = np.mean(np.abs(autoreg_res), axis = 1)
        amplitude = np.arange(lags) + 1
        autoreg_res = -amplitude / np.log(np.abs(autoreg_res))
        # # TODO: take out some significantly large timescale
        # extreme_index = np.where(autoreg_res > 1)
        # autoreg_res[extreme_index] = 0
        # region_timescale.append(np.max(autoreg_res))
        region_timescale.append(np.median(autoreg_res))
        regions_res[region] = autoreg_res
        regions_model[region] = all_models
    # Plot the timescale
    region_timescale = np.array(region_timescale)
    print("Region Timescale: ", region_timescale)
    sort_index = np.argsort(region_timescale)
    plt.figure(figsize=(7, 10))
    plt.title("Trial-Level Seasonal Timescale", fontsize = 20)
    plt.plot(region_timescale[sort_index], "bo--", lw = 3, ms = 10)
    plt.xticks(np.arange(len(region_timescale)), np.array(["reset", "show", "choose", "reward"])[sort_index],
               fontsize = 20, rotation = 30)
    plt.yticks(fontsize = 10)
    plt.ylabel("Timescale", fontsize = 20)
    # plt.ylim(0.5, 1.1)
    plt.show()
    return regions_model, regions_res


def regionBlockSeasonalAnalysis(data, lags, time_step_category):
    '''
    Intrinsic timescale analysis.
    :param data: Firing data with the shape of (block num, trial num in a block, time steps, neuron num) 
    :param lags: Time step lags for auto-regressive.
    :return: Fitted auto-regressive models for every neuron. The auto-regressive coefficients.
        all_models: A list with each element a fittted auto-regressive model (statsmodels.tsa.ar_model.AutoReg).
        autoreg_res: A matrix containing auto-regressive coefficient of all the neurons with the shape of (neuron num, lags). 
    '''
    data_shape = data.shape
    neurons_num = data_shape[-1]
    trial_num = data_shape[1]
    firing_data = np.zeros(
        (data_shape[0] * trial_num, neurons_num)
    )
    for i in range(neurons_num):
        trial_mean_firing_rate = np.mean(data[:, :, :, i], axis=2)
        firing_data[:, i] = trial_mean_firing_rate.reshape(-1)
    print("Data shape:", firing_data.shape)
    # ====================================
    #         Different Region
    # ====================================
    neuron_region = cateogorizeNeuron(data, time_step_category, need_plot = False)
    regions_res = {}
    regions_model = {}
    region_timescale = []
    for region in neuron_region:
        neurons_index = neuron_region[region]
        neurons_num = len(neurons_index)
        # Auto-regressive
        autoreg_res = np.zeros((neurons_num, lags))  # the auto-regressive coefficients for each neuron and time step
        all_models = []
        region_data = firing_data[:, neurons_index]
        for i in range(neurons_num):
            neuron_data = region_data[:, i]
            # filter out useless neurons
            if np.all(0 == neuron_data):
                continue
            model = AutoReg(neuron_data, lags=lags).fit()
            autoreg_res[i, :] = np.abs(model.params[1:])
        all_models.append(model)
        amplitude = np.arange(lags) + 1
        autoreg_res = -amplitude / np.log(np.abs(autoreg_res))
        # # TODO: take out some significantly large timescale
        # extreme_index = np.where(autoreg_res > 1)
        # autoreg_res[extreme_index] = 0
        # region_timescale.append(np.max(autoreg_res))
        region_timescale.append(np.median(autoreg_res))
        regions_res[region] = autoreg_res
        regions_model[region] = all_models
    # Plot the timescale
    region_timescale = np.array(region_timescale)
    print("Region Timescale: ", region_timescale)
    sort_index = np.argsort(region_timescale)
    plt.figure(figsize=(7, 10))
    plt.title("Block-Level Seasonal Timescale", fontsize = 20)
    plt.plot(region_timescale[sort_index], "bo--", lw = 3, ms = 10)
    plt.xticks(np.arange(len(region_timescale)), np.array(["reset", "show", "choose", "reward"])[sort_index],
               fontsize = 20, rotation = 30)
    plt.yticks(fontsize = 10)
    plt.ylabel("Timescale", fontsize = 20)
    # plt.ylim(0.0, 2.0)
    plt.show()
    return regions_model, regions_res


# ==================================================================
#                   BEHAVORIAL TIMESCALE　ANALYSIS
# ==================================================================
def behaveTimescale(rewards, choices):
    pass

# ==================================================================
#                   INTEGRATED MODEL ESTIMATION
# ==================================================================

def integratedEstimation(data, choices, rewards):
    '''
    Firing rate estimation with the integrated model including intrinsic, trial-level seasonal, reward, and choice 
    timescale (auto-regressive coefficient). Notes: block-level seasonal timescale is not-used here.
    :param data: All the firing rate data.
    :param choices: All choices.
    :param rewards: All rewards.
    :return: The estimation error.
    '''
    # Read in pre-computed parameters
    intrinsic_AR_coeff = np.load("intrinsic_AR_coeff.npy")
    trial_seasonal_AR_coeff = np.load("trial_seasonal_AR_coeff.npy")
    choice_AR_coeff = np.load("choice_AR_coeff.npy")
    reward_AR_coeff = np.load("reward_AR_coeff.npy")
    # Pre-processing
    rewards = rewards.reshape(-1)
    choices = choices.reshape(-1)
    for index in range(len(choices) - 1):
        if choices[index+1] > 3:
            choices[index+1] = choices[index]
    for index in range(len(rewards) - 1):
        if rewards[index + 1] > 3:
            rewards[index + 1] = rewards[index]
    # Reshape (average all the neurons)
    original_shape = data.shape
    data = data.reshape((-1, original_shape[2], original_shape[3]))
    data = np.nanmean(data, axis = 2) # (#trials, #time steps)
    time_step_mean = np.nanmean(data, axis = 0)
    intrinsic_AR_coeff = np.nanmean(intrinsic_AR_coeff, axis = 0) # (#lags)
    trial_seasonal_AR_coeff = np.nanmean(trial_seasonal_AR_coeff, axis = 0) # (#time step, #lags)
    # The number of lags for every AR
    intrinsic_lags = intrinsic_AR_coeff.shape[0]
    trial_seasonal_lags = trial_seasonal_AR_coeff.shape[1]
    choice_lags = choice_AR_coeff.shape[0]
    reward_lags = reward_AR_coeff.shape[0]
    # Construct the dataset (training, testing)
    trial_num = data.shape[0]
    time_step_num = data.shape[1]
    X = np.zeros((np.prod(data.shape), 4))
    Y = data.reshape(-1)
    # Fitting firing rate with AR components
    # TODO: carefully check this
    for trial_index in range(trial_num):
        for time_index in range(time_step_num):
            sample_index = trial_index * time_step_num + time_index
            # intrinsic component
            history_index = sample_index if sample_index <= intrinsic_lags else intrinsic_lags
            X[sample_index, 0] = intrinsic_AR_coeff[:history_index] @ Y[sample_index - history_index: sample_index] \
                if 0 != history_index else 0
            # trial-level seasonal component
            history_index = trial_index if trial_index <= trial_seasonal_lags else trial_seasonal_lags
            X[sample_index, 1] = trial_seasonal_AR_coeff[time_index, :history_index] @ data[trial_index - history_index: trial_index, time_index] \
                if 0 != history_index else 0
            # reward component
            history_index = trial_index if trial_index <= reward_lags else reward_lags
            X[sample_index, 2] = (reward_AR_coeff[:history_index] @ rewards[trial_index - history_index:trial_index]) * time_step_mean[time_index] \
                if 0 != history_index else 0
            # choice component
            history_index = trial_index if trial_index <= choice_lags else choice_lags
            X[sample_index, 3] = (choice_AR_coeff[:history_index] @ choices[trial_index - history_index:trial_index]) * time_step_mean[time_index]\
                if 0 != history_index else 0
    time_step_mean = np.tile(time_step_mean.reshape((-1, 1)), trial_num).T
    Y = Y - time_step_mean.reshape(-1)
    # Fit the firing rate
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    LR_coeff = np.abs(model.coef_) / np.sum(np.sum(model.coef_))
    print("Timescale coefficient: ", LR_coeff)
    # print("$R^2$: ", model.score(X_test, Y_test))
    Y_pred = model.predict(X_test)
    print("Mean Y value:", np.nanmean(np.abs(Y_test[np.where(Y_test != 0)])))
    print("MSE: ", mean_squared_error(Y_test, Y_pred))



# ==================================================================
#                   SAVE ANALYSIS RESULT FOR INTEGRATING MODEL
# ==================================================================

def basicAnalyisAndSave(all_firing_rate, choices, rewards):
    # Intrinsic timescale analysis
    print("\n", "=" * 10, " INTRINSIC ", "=" * 10)
    intrinsic_lags = 45
    all_models, intrinsic_res = intrinsicAnalysis(all_firing_rate, lags=intrinsic_lags)
    with open("intrinsic_AR_coeff.npy", 'wb') as file:
        np.save(file, intrinsic_res)
    print('Finished saving!')

    # Trial-level seasonal timescale analysis
    print("\n", "=" * 10, " TRIAL SEASONAL ", "=" * 10)
    trial_level_seasonal_lags = 5
    all_models, trial_seasonal_res = trialLevelSeasonalAnalysis(all_firing_rate, lags=trial_level_seasonal_lags)
    with open("trial_seasonal_AR_coeff.npy", 'wb') as file:
        np.save(file, trial_seasonal_res)
    print('Finished saving!')

    # Block-level seasonal timescale analysis
    print("\n", "=" * 10, " BLOCK SEASONAL ", "=" * 10)
    block_level_seasonal_lags = 5
    all_models, block_seasonal_res = blockLevelSeasonalAnalysis(all_firing_rate, lags=block_level_seasonal_lags)
    with open("block_seasonal_AR_coeff.npy", 'wb') as file:
        np.save(file, block_seasonal_res)
    print('Finished saving!')

    # Choice AR analysis
    choiceARLag = 10
    print("\n", "=" * 10, " CHOICE AR ", "=" * 10)
    model, choice_res = choiceARAnalysis(choices, lags=choiceARLag)
    with open("choice_AR_coeff.npy", 'wb') as file:
        np.save(file, choice_res)
    print('Finished saving!')

    # Reward AR analysis
    rewardARLag = 10
    print("\n", "=" * 10, " REWARD AR ", "=" * 10)
    model, reward_res = rewardARAnalysis(rewards, lags=rewardARLag)
    with open("reward_AR_coeff.npy", 'wb') as file:
        np.save(file, reward_res)
    print('Finished saving!')





if __name__ == '__main__':
    config = "ThreeArmed-Old"
    # Pre-processing
    if config == "ThreeArmed-Old":
        path = "RewardAffectData-OldTraining-OldNetwork-Three-Armed-Bandit/"
        log_file_name = path + "RewardAffectData-OldTraining-OldNetwork-ThreeArmed-sudden-reverse-model1-validation-1e6.hdf5"
    elif config == "TwoArmed":
        path = "Two-Armed-Bandit-SlowReverse/"
        log_file_name = path + "SimplifyTwoArmedSlowReverseNoNoise-validation-15e6.hdf5"
    else:
        raise ValueError("Undefined task name!")
    all_firing_rate, choices, rewards =getFiringRate(log_file_name)
    print("Firing data shape is ", all_firing_rate.shape)
    print("Finished preprocessing!")

    # # Intrinsic timescale analysis
    # print("\n", "="*10, " INTRINSIC ","="*10)
    # intrinsic_lags = 45
    # all_models, autoreg_res = intrinsicAnalysis(all_firing_rate, lags = intrinsic_lags)
    # plotIntrinsicResult(all_models, autoreg_res, intrinsic_lags)

    # # Trial-level seasonal timescale analysis
    # print("\n", "="*10, " TRIAL SEASONAL ","="*10)
    # trial_level_seasonal_lags = 5
    # all_models, autoreg_res = trialLevelSeasonalAnalysis(all_firing_rate, lags = trial_level_seasonal_lags)
    # plotTrialSeasonalResult(all_models, autoreg_res, trial_level_seasonal_lags)

    # # Block-level seasonal timescale analysis
    # print("\n", "="*10, " BLOCK SEASONAL ","="*10)
    # block_level_seasonal_lags = 5
    # all_models, autoreg_res = blockLevelSeasonalAnalysis(all_firing_rate, lags=block_level_seasonal_lags)
    # plotBlockSeasonalResult(all_models, autoreg_res, block_level_seasonal_lags)

    # # Choice memory timescale analysis
    # res = choiceMemoryAnalysis(all_firing_rate, choices)
    # print(res.summary())

    # # Choice AR analysis
    # choiceARLag = 10
    # print("\n", "=" * 10, " CHOICE AR ", "=" * 10)
    # model, autoreg_res = choiceARAnalysis(choices, lags = choiceARLag)
    # print(model.summary())
    # plotChoiceARResult(model, autoreg_res, lags = choiceARLag)

    # # Reward AR analysis
    # rewardARLag = 10
    # print("\n", "=" * 10, " REWARD AR ", "=" * 10)
    # model, autoreg_res = rewardARAnalysis(rewards, lags=rewardARLag)
    # print(model.summary())
    # plotRewardARResult(model, autoreg_res, lags=rewardARLag)

    # # Categorize neurons
    # cateogorizeNeuron(all_firing_rate, time_step_category = [[0,1, 9], [2,3,4], [5,6], [7,8]])


    # # Region intrinsic timescale analysis
    # print("\n", "="*10, " REGION INTRINSIC ","="*10)
    # intrinsic_lags = 20
    # all_models, autoreg_res = regionIntrinsicAnalysis(
    #     all_firing_rate,
    #     time_step_category = [[0,1, 9], [2,3,4], [5,6], [7,8]],
    #     lags = intrinsic_lags
    # )
    #
    # # Region trial seasonal timescale analysis
    # print("\n", "="*10, " REGION TRIAL SEASONAL ","="*10)
    # trial_seasonal_lags = 5
    # all_models, autoreg_res = regionTrialSeasonalAnalysis(
    #     all_firing_rate,
    #     time_step_category = [[0,1, 9], [2,3,4], [5,6], [7,8]],
    #     lags = trial_seasonal_lags
    # )
    #
    # # Region trial seasonal timescale analysis
    # print("\n", "=" * 10, " REGION TRIAL SEASONAL ", "=" * 10)
    # block_seasonal_lags = 5
    # all_models, autoreg_res = regionBlockSeasonalAnalysis(
    #     all_firing_rate,
    #     time_step_category=[[0, 1, 9], [2, 3, 4], [5, 6], [7, 8]],
    #     lags=block_seasonal_lags
    # )

    # # Integrated model firing rate analysis
    # basicAnalyisAndSave(all_firing_rate, choices, rewards)
    print("\n", "=" * 10, " INTEGRATED MODEL ", "=" * 10)
    integratedEstimation(all_firing_rate, choices, rewards)