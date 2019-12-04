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
from TorchNetwork import TorchNetwork
from net_tools import np2tensor, tensor2np, match_rate
from net_tools import readConfigures #TODO: change this
from ThreeArmedDataProcessor import ThreeArmedDataProcessor
from ThreeArmedValidateLogWriter import ThreeArmedValidateLogWriter


class ThreeArmedTask(Task):
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
        self.model = TorchNetwork(config_file) # TODO: change to initialize with dict rather than the file
        cofig_pars = readConfigures(config_file)
        self.data_helper = ThreeArmedDataProcessor(cofig_pars['data_file'], cofig_pars['validation_data_file'])
        np.random.seed()

    def train(self):
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
            train_loss, train_correct_rate = self.model.training(self.train_data, self.train_guide)
        except KeyboardInterrupt:
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
            'trial_length':14,
            'validate_trial_length':5,
            'block_size':150,
            'input_dim':10,
            'reward_prob':self.validate_data_attr['reward_probability'][0][0],
            'choice_step' : [5,6], # time steps of choosing  stimulus
            'reward_step'  : [7,8], # time steps of getting reward
            'choice_index' : [5,6,7], # index of choices in the input
            'reward_index' : [8,9] # index of reward in the input
        }
        wining_counts, trial_counts = 0, 0
        total_loss, total_correct_rate = 0, 0
        t_interval = 1000

        # Create log writer
        neuron_shape = list(map(lambda x: int(x), list(self.model._initHidden().data.shape))) # shape of network neurons
        behavior_shape = list(self.validate_data[0].shape) # shape of an input
        behavior_shape.pop(0)
        if need_log:
            self.log_writer = ThreeArmedValidateLogWriter(log_filename) #TODO: what if don't need to record
            self.log_writer.craeteHdf5File({'behavior_shape':behavior_shape, 'neuron_shape':neuron_shape})

        # Validating
        for step, trial in enumerate(self.validate_data):
            if step >= 1e800:
                break

            self._resetTrialRecords(self.task_validate_attr['reward_prob'][:,(step % 2*self.task_validate_attr['block_size'])]) # TODO: get the reward probability for this trial
            tmp_loss, tmp_correct_rate, raw_rec, wining = self._trialValidate(trial)
            trial_counts = trial_counts + 1
            wining_counts = wining_counts + wining
            # completed_counts = completed_counts + completed
            total_loss = total_loss + tmp_loss
            total_correct_rate = total_correct_rate + tmp_correct_rate

            # Write into log file
            if need_log:
                self.log_writer.appendRecord(raw_rec)

            # Print out intermediate validating result: loss and correct rate
            if (step + 1) % t_interval == 0:
                print(
                    "(Validate)STEP: {:6d} | AVERAGE CORRECT RATE: {:6f} | AVERAGE LOSS: {:.6f} ".format(
                        step + 1,
                        total_correct_rate / t_interval,
                        total_loss / t_interval))
                total_loss = 0
                total_correct_rate = 0
            # Finish the validation but less than t_interval trials are left
            if step == (len(self.validate_data) - 1):
                left_trial_num = (step + 1) % t_interval
                print(
                    "(Validate)STEP: {:6d} | AVERAGE CORRECT RATE: {:6f} | AVERAGE LOSS: {:.6f} ".format(
                        step + 1,
                        total_correct_rate / left_trial_num,
                        total_loss / left_trial_num))

        # Clean up
        print('-'*40)
        print('Winning rate : {}'.format(wining_counts / trial_counts))
        if need_log:
            self.log_writer.closeHdf5File()

    def _trialValidate(self, trial):
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
        #self._resetTrialRecords()
        self.validate_records['trial_end'] = False
        self.validate_records['trial_pool'][:self.task_validate_attr['validate_trial_length'],:] = trial

        cur_time_step = self.validate_records['trial_pool'][self.task_validate_attr['validate_trial_length']-1,:]
        while not self.validate_records['trial_end']:
            next_time_step = self._estimateNextTimeStep(cur_time_step)
            cur_time_step = next_time_step

        # Convert tensor to numpy
        predicted_trial = np.around(self.validate_records['raw_records']).astype(np.int)

        raw_rec["sensory_sequence"] = copy.deepcopy(np.array(self.validate_records['sensory_sequence']).squeeze())  # validation set
        raw_rec["predicted_trial"] = copy.deepcopy(predicted_trial)  # predicted set
        raw_rec["raw_records"] = copy.deepcopy(self.validate_records['raw_records'])  # raw output
        raw_rec["hidden_records"] = copy.deepcopy(self.validate_records['hidden_records'])  # raw hidden
        raw_rec['choice'] = self.validate_records['choice']
        raw_rec['reward'] = 1- self.validate_records['reward'] # in records, 0 stands for reward wihle 1 stands for no reward
        # Compute correct rate; when calculate correct ratio, we need to truncate the trial
        sensory_sequence = raw_rec["sensory_sequence"]
        raw_records = raw_rec["raw_records"]
        tmp_correct_rate = match_rate(sensory_sequence[1:, :], predicted_trial[:-1, :])
        tmp_loss = self._MSELoss(sensory_sequence[1:], raw_records[:-1], sensory_sequence.shape[0]*sensory_sequence.shape[1])
        wining = 1 - self.validate_records['reward'].item() # in records, 0 stands for reward wihle 1 stands for no reward
        # completed = 1 if self.validate_records['completed'] else 0 TODO: don't need completed
        return tmp_loss, tmp_correct_rate, raw_rec, wining

    def _estimateNextTimeStep(self, cur_time_step):
        '''
        Estimate features at the next time step given features of the current time step.
        :param cur_time_step: Features of current time step.
        :return: 
        '''
        self.validate_records['time_step'] += 1
        hidden = self.validate_records['hidden']
        cur_time_step = torch.tensor(cur_time_step).clone().detach().type(torch.FloatTensor)\
            .view(1, self.model.network.batch_size,-1).requires_grad_(True)
        output, hidden = self.model.network(cur_time_step, hidden)
        # Determine whether need to update the trial
        if self.validate_records['time_step'] >= self.task_validate_attr['validate_trial_length']:
            next_time_step = self._updteTrial(self.validate_records['time_step'], output.detach().numpy())
        else:
            next_time_step = cur_time_step
        # Update validation records
        self.validate_records['hidden'] = hidden
        self.validate_records['raw_records'].append(tensor2np(output).reshape(-1))
        self.validate_records['hidden_records'].append(tensor2np(hidden))
        self.validate_records['sensory_sequence'].append(tensor2np(cur_time_step))
        if self.validate_records['time_step'] == self.task_validate_attr['trial_length']:
            self.validate_records['trial_end'] = True
        return next_time_step

    def _updteTrial(self, time_step_index, time_step_input):
        '''
        From the raw output to compute the binary value output for the next time step.
        :param time_step_index: 
        :param time_step_input: The input for the current time step with the shape of (1, input dimension)
        :return: 
        '''
        # If need to choose stimulus at this time step
        if time_step_index in self.task_validate_attr['choice_step']:
            if time_step_index == self.task_validate_attr['choice_step'][0]:
                action_options = time_step_input[0, self.task_validate_attr['choice_index']]
                # action_options = action_options.cpu().detach().numpy() if self.model.cuda_enabled else action_options.detach().numpy()
                pro_soft = self._softmax(action_options)
                idx = torch.tensor(np.random.choice(pro_soft.size, 1, p=pro_soft))  # choose a stimulus depends on the estimated probability
                # idx = torch.tensor(np.argmax(pro_soft))
                selected_stimulus = idx.data.item()  # 0 is choose A; 1 is choose B; 2 is choose C
                time_step_input = np.round(time_step_input)
                time_step_input[:, self.task_validate_attr['choice_index']] = 0
                time_step_input[:, self.task_validate_attr['choice_index'][0]+selected_stimulus] = 1
                self.validate_records['choice'] = selected_stimulus
            else:
                time_step_input = np.round(time_step_input)
                time_step_input[:, self.task_validate_attr['choice_index']] = 0
                time_step_input[:, self.task_validate_attr['choice_index'][0] + self.validate_records['choice']] = 1
                self.validate_records['trial_pool'][time_step_index, :] = time_step_input
        # If need to get reward at this time step
        elif time_step_index in self.task_validate_attr['reward_step']:
            if time_step_index == self.task_validate_attr['reward_step'][0]:
                reward = self.validate_records['trial_reward_prob'][self.validate_records['choice']]
                is_reward = np.random.choice([0, 1], 1, p=(reward, 1-reward)) # 0 for reward, 1 for not reward
                time_step_input = np.round(time_step_input)
                time_step_input[:, self.task_validate_attr['reward_index']] = 0
                time_step_input[:, self.task_validate_attr['reward_index'][0]+is_reward] = 1
                self.validate_records['reward'] = is_reward # in the records, also 0 for reward while 1 for not reward
            else:
                time_step_input = np.round(time_step_input)
                time_step_input[:, self.task_validate_attr['reward_index']] = 0
                time_step_input[:, self.task_validate_attr['reward_index'][0] + self.validate_records['reward']] = 1
        # No specific requirements
        else:
            time_step_input = np.round(time_step_input)
        self.validate_records['trial_pool'][time_step_index, :] = time_step_input
        return time_step_input

    def _softmax(self,x, beta=8):
        '''
        Softmax function: np.exp(beta * x) / np.sum(np.exp(beta * x))
        :param x: Input of softmax functin.
        :param beta: Beta.
        :return: 
            Sofmax function value.
        '''
        return np.exp(beta * x) / np.sum(np.exp(beta * x), axis=0)

    def _MSELoss(self, estimation, true_value, sample_num):
        return np.sum(np.power(np.array(estimation - true_value), 2)) / sample_num

    def _resetTrialRecords(self, trial_reward_prob):
        '''
        Reset validating records.
        :return: VOID
        '''
        self.validate_records = {'raw_records': [],
                                 'hidden_records': [],
                                 'sensory_sequence': [],
                                 'time_step': -1,
                                 'trial_pool': np.zeros(
                                     (self.task_validate_attr['trial_length']+1, self.task_validate_attr['input_dim']) #TODO: add one time step here, pay attention to the evaluation
                                 ),
                                 'trial_reward_prob':trial_reward_prob,
                                 'completed': None,
                                 'trial_end': False,
                                 'block': [],
                                 'reward': None,
                                 'choice': None,
                                 'chosen': None,
                                 'state': None,
                                 'hidden': self.model._initHidden()
                                 }

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
    reward_type = 'sudden-reverse'
    config_file = "ThreeArmed_Config.json"
    configs = readConfigures(config_file)
    model_name = './save_m/model-three-armed-'+ configs['data_file'].split('-')[2] + '-' + reward_type +'.pt' # TODO: deal with multiple models

    t = ThreeArmedTask(config_file)
    # t.train()
    # t.saveModel(model_name)
    t.loadModel('./save_m/model-three-armed-2019_12_04-sudden-reverse-without_init.pt', 'ThreeArmed_Config.json')
    t.validate('validate_record-three-armed-2019_12_04-sudden-reverse-without_init.hdf5')