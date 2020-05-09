''''
ThreeArmedTask.py: Implement the three-armed bandit task, including the training and validating processes.

Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Nov. 29 2019
'''
import copy
import numpy as np
import torch
import sys

sys.path.append('../Network/')
from Task import Task
# from StackedGRUNetwork import StackedGRUNetwork
from TorchNetwork import TorchNetwork
from net_tools import np2tensor, tensor2np, match_rate
from net_tools import readConfigures
from ThreeArmedDataProcessor import ThreeArmedDataProcessor #TODO: class name
from ThreeArmedValidateLogWriter import ThreeArmedValidateLogWriter #TODO: class name


class ThreeArmedTask(Task): #TODO: change the class name to two-armed task, so as other class
    '''
    Description:
        The three-armed bandit task, including training and validating processes.
    
    Variables:
        ----------- ALWAYS EXIST ----------------
        model: The GRU network model.
        data_helper: An instance of ``DataProcessor''. Used for preparing training and validating datasets.
        ------------ EXIST AFTER TRAINING -------
        train_data: Training datasset.
        train_guide: The weight coefficients for each element of training trial.
        ------------ EXIST AFTER VALIDATING -----
        validate_data: Validating dataset.
        validate_data_attr: Settings for validating.
        validate_record: Validating records.
        log_writer: Used for writing log into hdf5 files.
        
    Functions:
        __init__: Initialize the task.
        train: Training.
        validate: Validating.
        saveModel: Save the trainied GRU network model.
        loadModel: Load an existing trained GRU network model.
        _nextInput: Simulate the input of next time step given the current action.
        _estimateAction: Estimate the action of a trial.
        _softmax: Softmax function.
        _resetTrialRecords: Reset validating records.
        _trialValidate: Do validate for a trial.
    '''

    def __init__(self, config_file):
        '''
        Initialize the task.
        :param config_file: Configuration file. Should be a JSON file.
        '''
        super(ThreeArmedTask, self).__init__()
        # TODO: change to initialize with dict rather than the file
        self.model = TorchNetwork(config_file)
        cofig_pars = readConfigures(config_file)
        self.data_helper = ThreeArmedDataProcessor(cofig_pars['data_file'], cofig_pars['validation_data_file'])
        np.random.seed()

    def train(self, save_iter = 0): #TODO: save iteration
        '''
        Training.
        :return: 
            train_loss(ndarray): The training loss.
            train_correct_rate(ndarray): The training correct rate.
        '''
        print('='*40)
        print('START TRAINING...')
        self.train_data, self.train_guide = self.data_helper.prepareTrainingData()
        try:
            train_loss, train_correct_rate = self.model.training(self.train_data, self.train_guide, save_iter = save_iter)
        except KeyboardInterrupt: #TODO: check this
            self.model.trained = True
            self.saveModel('interrupted_model.pt')
            return
        return train_loss, train_correct_rate

    def validate(self, log_filename = ''):
        '''
        Validating.
        :return: 
        '''
        if not self.model.trained:
            raise ValueError('Untrained network: the network should be trained first or read from a .pt file!')
        if log_filename == '':
            need_log = False
        else:
            need_log = True
        print('=' * 40)
        print('START VALIDATING...')

        # Initialization
        self.validate_data, self.validate_data_attr = self.data_helper.prepareValidatingData()
        self.task_validate_attr = { # attributes for validating
            'trial_length':9,
            # 'block' : self.validate_data_attr['block'],
            # 'trans_probs' : self.validate_data_attr['trans_probs'][0][0],
            'reward_prob_1' : self.validate_data_attr['reward_prob_1'][0][0],
            'action_step' : [5,6],
            'about_state'  : [0,1,2], # index of showing stimulus
            'about_choice' : [4,5,6,7], # index of choosing choices
            'about_reward' : [8,9], # index of reward
            'interrupt_states' : [[0, 0, 0, 1, 1, 0, 0, 0, 0, 1]],
            'chosen_states' : [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
                ],
            'hidden':self.model._initHidden()
        }
        wining_counts, completed_counts, trial_counts = 0, 0, 0
        total_loss, total_correct_rate = 0, 0
        t_interval = 1000

        # Create log writer
        neuron_shape = list(map(lambda x: int(x), list(self.model._initHidden().data.shape)))
        behavior_shape = list(self.validate_data[0].shape)
        behavior_shape.pop(0)
        if need_log:
            self.log_writer = ThreeArmedValidateLogWriter(log_filename)
            self.log_writer.craeteHdf5File({'behavior_shape':behavior_shape, 'neuron_shape':neuron_shape})

        # Validating
        for step, trial in enumerate(self.validate_data):
            if step >= 1e800:
                break
            trial_reward_prob = {
                # 'trans_probs': self.task_validate_attr['trans_probs'][:,step],
                'reward_prob_1':self.task_validate_attr['reward_prob_1'][:,step]
            }
            # self._resetTrialRecords()
            tmp_loss, tmp_correct_rate, raw_rec, wining, completed = self._trialValidate(trial, trial_reward_prob)
            trial_counts = trial_counts + 1
            wining_counts = wining_counts + wining
            completed_counts = completed_counts + completed
            total_loss = total_loss + tmp_loss
            total_correct_rate = total_correct_rate + tmp_correct_rate

            #TODO: for printing correct rate
            raw_rec['correct_rate'] = tmp_correct_rate

            # Write into log file
            if need_log:
                self.log_writer.appendRecord(raw_rec)

            # Print out intermediate validating result: loss and correct rate
            if (step + 1) % t_interval == 0 or step == (len(self.validate_data) - 1):
                print(
                    "(Validate)STEP: {:6d} | AVERAGE CORRECT RATE: {:6f} | AVERAGE LOSS(without loss): {:.6f} ".format(
                        step + 1,
                        total_correct_rate / t_interval,
                        total_loss / t_interval))
                total_loss = 0
                total_correct_rate = 0

        # Clean up
        print('-'*40)
        print('Winning rate : {},  Completed rate : {}'.format(wining_counts / trial_counts, completed_counts / trial_counts))
        if need_log:
            self.log_writer.closeHdf5File()

    def _trialValidate(self, trial, trial_reward_prob):
        '''
        Validating a trial.
        :param trial: The trial.
        :param trial_reward_prob: Reward probability of the trial. 
        :return: 
            tmp_loss(float): Loss of validating this trial.
            tmp_correct_rate(float): Correct rate of validating this trial.
            raw_rec(dict): A dict of trial validating records with four keys "sensory_sequence", "predicted_trial", 
                          "raw_records", and "hidden_records".
            wining(int): Count of wining trials.
            completed(int): Count of completed trials.
        '''
        raw_rec = {}
        self._resetTrialRecords()
        self.validate_records.update({'states_pool': trial.tolist(), 'trial_end': False})
        action = [0]  # init action: fixate on fixation point [Jiaqi] 2 for fixation
        planaction_record = []
        while not self.validate_records['trial_end']:
            trial_end, sensory_inputs = self._nextInput(action, trial_reward_prob)  # compute the input
            action = self._estimateAction(sensory_inputs)
            planaction_record.append(action)
        # Convert tensor to numpy
        predicted_trial = np.around(self.validate_records['raw_records']).astype(np.int)
        for i in range(predicted_trial.shape[0]):
            if len(planaction_record[i]) != 0:
                action = planaction_record[i][0]
                predicted_trial[i, self.task_validate_attr['about_choice']] = 0 #
                predicted_trial[i, self.task_validate_attr['about_choice'][0] + action] = 1
            elif len(planaction_record[i]) == 0:
                predicted_trial[i, self.task_validate_attr['about_choice']] = 0

        raw_rec["sensory_sequence"] = copy.deepcopy(self.validate_records['sensory_sequence'])  # validation set
        raw_rec["predicted_trial"] = copy.deepcopy(predicted_trial)  # predicted set
        raw_rec["raw_records"] = copy.deepcopy(self.validate_records['raw_records'])  # raw output
        raw_rec["hidden_records"] = copy.deepcopy(self.validate_records['hidden_records'])  # raw hidden

        if self.validate_records['reward'] != None:
            raw_rec['choice'] = self.validate_records['choice']
            raw_rec['reward'] = self.validate_records['reward']
        else:
            raw_rec['choice'] = -1
            raw_rec['reward'] = -1

        # Compute correct rate; when calculate correct ratio, we need to truncate the trial
        tmp_correct_rate = match_rate(np.array(self.validate_records['sensory_sequence'])[1:, :],
                                      predicted_trial[:-1, :])
        # TODO: put computing loss into a function
        tmp_loss = np.sum(np.power(
            np.array(self.validate_records['sensory_sequence'])[1:] - np.array(self.validate_records['raw_records'])[
                                                                      :-1], 2)) / (
                       np.array(self.validate_records['sensory_sequence']).shape[0] *
                       np.array(self.validate_records['sensory_sequence']).shape[1])
        wining = 1 if self.validate_records['reward'] else 0
        completed = 1 if self.validate_records['completed'] else 0
        return tmp_loss, tmp_correct_rate, raw_rec, wining, completed

    def _nextInput(self, action, trial_reward_prob):
        '''
        Simulate the input of next time step given the current action.
        :param action: The current action.
        :param trial_reward_prob: Reward probability for this trial.
        :return: 
            trial_end(boolean): Whether this trial is end. 
            next_sensory_inputs: Next input of this trial
        :raise BaseException: Wrong time index.
        '''
        time_step = self.validate_records['time_step'] + 1
        states_pool = self.validate_records['states_pool']
        if len(action) != 1 or not (action[0] in (0, 1, 2, 3)):
            states_pool = copy.deepcopy(self.task_validate_attr['interrupt_states'])
        else:
            action = action[0]
            if not (time_step in self.task_validate_attr['action_step']):  # choice has not been made
                if action == 0:  # fixate, keep silent
                    pass
                else:
                    states_pool = copy.deepcopy(self.task_validate_attr['interrupt_states'])

            elif time_step == self.task_validate_attr['action_step'][0]:  # choice has not been made
                if action == 0:  # fixate, keep silent
                    states_pool = copy.deepcopy(self.task_validate_attr['interrupt_states'])

                if action != 0: # action = 0 or 1, representing A or B respectively
                    # print("block: {:6f} | choice: {:6f} | reward: {:.6f} ".format(self.block,action,reward))
                    reward = trial_reward_prob['reward_prob_1'][action-1] > np.random.rand()
                    self.validate_records.update({
                        # 'common': state == action,
                        'choice':action,
                        'reward':reward,
                        'chose':True})
                    state_ = copy.deepcopy(self.task_validate_attr['chosen_states'])
                    state_[0][self.task_validate_attr['about_choice'][action]] = 1 # action is 1/2 and the index is 0/1
                    state_[1][self.task_validate_attr['about_choice'][action]] = 1

                    state_[0][self.task_validate_attr['about_state'][action-1]] = 1
                    state_[1][self.task_validate_attr['about_state'][action-1]] = 1

                    state_[2][self.task_validate_attr['about_reward'][1 - int(reward)]] = 1
                    state_[3][self.task_validate_attr['about_reward'][1 - int(reward)]] = 1
                    states_pool = copy.deepcopy(state_)

            elif time_step == self.task_validate_attr['action_step'][1]:  # choice has not been made
                if action == self.validate_records['choice']:
                    pass
                else:
                    states_pool = copy.deepcopy(self.task_validate_attr['interrupt_states'])
            else:
                raise BaseException("Wrong time step index!")
        try:
            next_sensory_inputs = states_pool.pop(0)
            trial_end = self.validate_records['trial_end']
            completed = self.validate_records['completed']
            reward = self.validate_records['reward']
            if len(states_pool) == 0:
                trial_end = True
                if time_step == self.task_validate_attr['trial_length']:
                    completed = True
                else:
                    reward = None
            self.validate_records.update({
                'states_pool':states_pool,
                'trial_end':trial_end,
                'completed':completed,
                'reward':reward,
                'time_step':time_step})
            return trial_end, next_sensory_inputs
        except IndexError:
            print("Wrong States Pool!")

    def _estimateAction(self,sensory_inputs):
        '''
        Estimate the action given trial inputs.
        :param sensory_inputs: Trial inputs.
        :return: 
            selected_action: The estimated action.
        '''
        hidden = self.task_validate_attr['hidden']
        processed_input = torch.tensor(sensory_inputs).type(torch.FloatTensor).view(1, self.model.network.batch_size, -1).requires_grad_(True)
        output, hidden = self.model.network(processed_input, hidden)
        action_options = output[0, self.task_validate_attr['about_choice']]
        # TODO: what if the model is in the GPU but is validated in CPU
        action_options = action_options.cpu().detach().numpy() if self.model.cuda_enabled else action_options.detach().numpy()

        pro_soft = self._softmax(action_options)
        idx = torch.tensor(np.random.choice(pro_soft.size, 1, p=pro_soft))
        selected_action = [idx.data.item()]  # 0: fixation point, 1: left, 2: right, 3: other position
        # update validation records
        self.task_validate_attr['hidden'] = hidden
        self.validate_records['raw_records'].append(tensor2np(output).reshape(-1))
        self.validate_records['hidden_records'].append(tensor2np(hidden))
        self.validate_records['sensory_sequence'].append(sensory_inputs)
        return selected_action

    def _softmax(self,x, beta=8):
        '''
        Softmax function: np.exp(beta * x) / np.sum(np.exp(beta * x))
        :param x: Input of softmax functin.
        :param beta: Beta.
        :return: 
            Sofmax function value.
        '''
        return np.exp(beta * x) / np.sum(np.exp(beta * x), axis=0)

    def _resetTrialRecords(self):
        '''
        Reset validating records.
        :return: VOID
        '''
        self.validate_records = {'raw_records': [],
                                 'hidden_records': [],
                                 'sensory_sequence': [],
                                 'time_step': -1,
                                 'states_pool': [],
                                 'completed': None,
                                 'trial_end': False,
                                 # 'block': [],
                                 'reward': None,
                                 'choice': None,
                                 'chosen': None,
                                 'state': None,
                                 # 'hidden': self.model._initHidden()
                                 }

    # def _MSELoss(self, estimation, true_value, sample_num):
    #     return np.sum(np.power(np.array(estimation - true_value), 2)) / sample_num

    def saveModel(self, filename):
        '''
        Save the trained model into a file.
        :param filename: Name of a Pytorch model file.  
        :return: VOID
        :raise ValueError: Save Model: the model should be trained before saving!
        '''
        if not self.model.trained:
            raise ValueError('Save Model: the model should be trained before saving!')
        self.model.saveModel(filename)

    def loadModel(self, filename, config_file):
        '''
        Load a Pytorch model from file.
        :param filename: Filename of a Pytorch model.
        :param config_file: Filename of the task conmfiguration.
        :return: VOID
        '''
        self.model.loadModel(filename, config_file)
        self.model.network.eval()
        self.model.trained = True


if __name__ == '__main__':
    torch.set_num_threads(3)
    torch.set_num_interop_threads(3)
    t = ThreeArmedTask('ThreeArmed_Config.json')

    # t.train(save_iter=100000)
    # t.saveModel('./save_m/'+'RewardAffectData-OldTraining-OldNetwork--ThreeArmed-1e6-model.pt')

    # t.loadModel('./save_m/sudden-reverse/model3/RewardAffectData-OldTraining-OldNetwork-ThreeArmed-1e6-model.pt', 'ThreeArmed_Config.json')
    t.loadModel('./save_m/slow-reverse/model3/RewardAffectData-OldTraining-OldNetwork-ThreeArmed-1e6-model.pt', 'ThreeArmed_Config.json')
    t.validate('RewardAffectData-OldTraining-OldNetwork-ThreeArmed-slow-reverse-model3-validation-1e6.hdf5')