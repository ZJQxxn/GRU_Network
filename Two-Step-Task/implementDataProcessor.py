import os
import copy
import datetime
import numpy as np
import scipy.io as sio
from DataProcessor import DataProcessor

class implementDataProcessor(DataProcessor):
    def __init__(self, trans_prob, reward_prob):
        '''
        Initialization.
        '''
        self.trans_prob = trans_prob
        self.reward_prob = reward_prob
    def prepareTrainingData(self,NumTrials,block_size,Double):
        '''
        Preparing training dataset.
        '''
        info = {'NumTrials': NumTrials, 'reward_prob': self.reward_prob, 'block_size': block_size, 'trans_prob': self.trans_prob}
        temp = np.hstack((np.ones((block_size,)), np.zeros((block_size,))))
        blocks = np.tile(temp, int(NumTrials / (block_size * 2)))
        lost_trialsNum = NumTrials - blocks.size
        if lost_trialsNum <= block_size:
            temp = np.ones((lost_trialsNum,))
        else:
            temp = np.hstack((np.ones((block_size,)), np.zeros((lost_trialsNum - block_size,))))
        blocks = np.hstack((blocks, temp))
        choices = np.random.choice([0, 1], NumTrials)  # 0:A1; 1:A2
        trans_probs = self.trans_prob * choices + (1 - self.trans_prob) * (
                    1 - choices)  # probability of transition to the B2 in stage 2

        temp = np.random.rand(NumTrials, )
        state2 = trans_probs > temp  # 0: B1; 1: B2

        reward_prob_B1 = (1 - self.reward_prob) * state2 * (1 - blocks) + self.reward_prob * (1 - state2) * (
                    1 - blocks)  # when B1 with larger reward
        reward_prob_B2 = self.reward_prob * state2 * blocks + (1 - self.reward_prob) * (
                    1 - state2) * blocks  # when B2 with larger reward

        reward_prob_state2 = reward_prob_B1 + reward_prob_B2  # reward probability of the observation in stage 2

        temp1 = np.random.rand(NumTrials, )
        reward_all = reward_prob_state2 > temp1
        state_all = copy.deepcopy(state2) + 1  # 1: B1; 2: B2

        data_ST = []
        n_input = 10;
        trial_length = 13
        shape_Dur = 3;  # period for shape presentation
        choice_Dur = 2;  # period for shape interval
        for nTrial in range(NumTrials):
            inputs = np.zeros((n_input, trial_length))
            inputs[0:2, 2:5] = 1  # the three-five time points representing the first epoch

            if choices[nTrial] == 0:
                inputs[6, 5:7] = 1
            elif choices[nTrial] == 1:
                inputs[7, 5:7] = 1

            if state_all[nTrial] == 1:
                inputs[2, 7:10] = 1
            elif state_all[nTrial] == 2:
                inputs[3, 7:10] = 1

            if reward_all[nTrial] == 1:
                inputs[8, 10:12] = 1

            inputs[4, :] = inputs[0:4, :].sum(axis=0)
            inputs[4, np.where(inputs[4, :] != 0)] = 1
            inputs[4, :] = 1 - inputs[4, :]
            inputs[5, :] = 1 - inputs[6:8, :].sum(axis=0)
            inputs[9, :] = 1 - inputs[8, :]
            if nTrial != 0:
                data_ST.append([np.hstack((inputs_prev, inputs)).T])
            inputs_prev = copy.deepcopy(inputs)

        if Double:
            training_guide = np.array(
                [reward_all[1:], 2 * trial_length + np.zeros((len(reward_all) - 1,))]).squeeze().astype(
                np.int).T.tolist()
        else:
            training_guide = np.array(
                [reward_all[1:], trial_length + np.zeros((len(reward_all) - 1,))]).squeeze().astype(np.int).T.tolist()

        data_ST_Brief = {'choices': choices, 'state_all': state_all,
                         'reward': reward_all, 'trans_prob': self.trans_prob,
                         'shape_Dur': shape_Dur, 'choice_Dur': choice_Dur, 'Double': Double,
                         'training_guide': training_guide}

        # data  saving
        pathname = "../data/"
        file_name = datetime.datetime.now().strftime("%Y_%m_%d")
        data_name = 'SimpTwo_TrainingSet-' + file_name

        n = 0
        while 1:  # save the model
            n += 1
            if not os.path.isfile(pathname + data_name + '-' + str(n) + '.mat'):
                sio.savemat(pathname + data_name + '-' + str(n) + '.mat',
                            {'data_ST': data_ST,
                             'data_ST_Brief': data_ST_Brief,
                             'info': info})
                print("_" * 36)
                print("training file for simplified two step task is saved")
                print("file name:" + pathname + data_name + '-' + str(n) + '.mat')
                break
        filename = pathname + data_name + '-' + str(n) + '.mat'

    def prepareTestingData(self,NumTrials,block_size):
        '''
        Prepare testing dataset.
        '''
        temp = np.hstack((np.ones((block_size,)), np.zeros((block_size,))))
        blocks = np.tile(temp, int(NumTrials / (block_size * 2)))
        lost_trialsNum = NumTrials - blocks.size
        if lost_trialsNum <= block_size:
            temp = np.ones((lost_trialsNum,))
        else:
            temp = np.hstack((np.ones((block_size,)), np.zeros((lost_trialsNum - block_size,))))
        blocks = np.hstack((blocks, temp))

        trans_probs = self.trans_prob * np.ones(NumTrials, )
        reward_probs = self.reward_prob * blocks + (1 - self.reward_prob) * (1 - blocks)

        inputs = [
            [0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0.]
        ]
        data_ST = [[np.array(inputs).T]] * NumTrials
        data_ST_Brief = {'reward_prob_1': reward_probs, 'trans_probs': trans_probs, 'block_size': block_size,
                         'block': blocks}
        pathname = "../data/"
        file_name = datetime.datetime.now().strftime("%Y_%m_%d")
        data_name = 'SimpTwo_TestingSet-' + file_name
        n = 0
        while 1:  # save the model
            n += 1
            if not os.path.isfile(pathname + data_name + '-' + str(n) + '.mat'):
                sio.savemat(pathname + data_name + '-' + str(n) + '.mat',
                            {'data_ST': data_ST,
                             'data_ST_Brief': data_ST_Brief,
                             })
                print("_" * 36)
                print("testing file for simplified two step task is saved")
                print("file name:" + pathname + data_name + '-' + str(n) + '.mat')
                break
        filename = pathname + data_name + '-' + str(n) + '.mat'
