'''
TorchNetwork.py: Implementation of the class for our gated neural unit (GRU) network with Pytorch.

Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Nov. 15 2019

Reference: 

'''
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append('./')
from net_tools import  readConfigures, validateConfigures
from net_tools import tensor2np, np2tensor, match_rate

import copy

class TorchNetwork:
    '''
    The GRU neural network for training and predicting. 
    
    Functions:
        __init__: Initialize the network with a configuration file. 
        training: Train the network.
        predicting: Predict outputs with the trained network.
        saveMiodel: Save network parameters and configurations to a Pytorch ``.pt''` file.
        loadModel: Read network parameters and configurations from a Pytorch 11.pt'' file.
        _trainBlock: Train the network with a data block.
        _trainTrial: Train the networkwith a single trial.
        _initHidden: Initialize the hidden units.
        _resetNetwork: Reset the network after creating an instance or loading model from file
        
    Variables:
        config_pars: A dict of all the network configurations.
        trained: Denote whether the network has been trained, either True or False.
        network: The three-layer GRU network.
        cude_enabled: Denote whether CUDA is available, either True or False.
        init_noise_amp: Coefficient of noise perturbation, when initializing the hidden units. Either True or False.
        rest_hidden: Denote whether the hidden units should be initialized between blocks, either True or False. 
        gradient_clip: Gradient clip. 
    '''

    def __init__(self, config_file = ""):
        '''
        Initialize the GRU network model. 
        :param config_file: Name of the configuration file, should be a JSON file name of srting type ("" by default).
        '''
        if config_file is not "": # TODO: init with a dict
            self.config_pars = readConfigures(config_file) # read configurations from file
            self.config_pars = validateConfigures(self.config_pars) # check validation of configurations
            self._resetNetwork() # initialize the network with a configuration file
        self.trained = False # denote whether the network has been trained

    def training(self, train_set, train_guide, truncate_iter = 1e800):
        '''
        Train the GRU  neural network.
        :param train_set: Training dataset, should have shape (number of trials, time steps in a trial, feature dimension)
        :param train_guide: Training reward, should have shape (number of trials, 2).
                                train_guide[:,0] determines whether this trial is used for training;
                                train_guide[:,1] determines how many first time steps in this trial is used. 
        :param truncate_iter: Truncated iteration. The network training stops when this number of trials are trained.
        :return: 
            - train_loss(list): Losses of every block.
            - train_correct_rate(list): Correct rates of every block.
        '''
        print('Reset hidden {}'.format(self.reset_hidden))
        batch_data = []
        batch_reward = []
        train_loss = []
        train_correct_rate = []
        batch_count = 1 # count the number of batches
        self.hidden = self._initHidden()
        # print("self.hidden:", self.hidden)
        for step, trial in enumerate(train_set):
            # cease training when reaching at the truncate iteration
            if step >= (truncate_iter - 1):
                break
            # Collect ``batch_size'' number of trials as training input
            batch_data.append(trial)
            batch_reward.append(train_guide[step])
            # training for every batch
            if (step + 1) % self.network.batch_size == 0:
                # reset hidden unit for next batch
                if self.reset_hidden:
                   self.hidden = self._initHidden()
                # train the network with current block of data
                batch_data = np.array(batch_data).transpose([1, 0, 2])
                batch_reward = np.array(batch_reward)
                batch_loss, batch_correct_rate = self._trainBatch(batch_data, batch_reward)
                batch_data,batch_reward=[],[]
                train_loss.append(batch_loss)
                train_correct_rate.append(batch_correct_rate)
                # TODO: print out or write into log file?
                print('=' * 15, " {}-th batch ".format(batch_count), '=' * 15)
                print("Network loss is : ", batch_loss)
                print("Correct rate is : ", batch_correct_rate)
                batch_count = batch_count + 1
            else:
                # Continue collecting training input trials
                continue
        self.trained = True
        return train_loss, train_correct_rate

    def predicting(self, test_set):
        '''
        Predict the output given testing dataset.
        :param test_set: Testing dataset with shape of (number of trials, dimensionality of input).
        :return: Predicted output with shape of (number of trials, dimensionality of output).
        '''
        if not self.trained:
            raise ValueError('The network has not been trained!')

        test_prediction = []
        for trial in test_set:
            trial_output = tensor2np(self.network(trial, self.hidden))
            test_prediction.append(trial_output)
        return np.array(test_prediction)

    def saveModel(self, filename):
        '''
        Save Pytorch network to a file with the configurations dict.
        :param filename: The filename of loaded network.
        :return: VOID
        '''
        pars = self.network.state_dict()
        torch.save(pars, filename)

    def loadModel(self, filename, config_file = ''):
        '''
        Load Pytorch network from .pt file, with all the network training configurations.
        :param filename: Filename of .py file.
        :return: VOID
        '''
        pars = torch.load(filename)
        self.config_pars = readConfigures(config_file)  # read configurations from file
        self.config_pars = validateConfigures(self.config_pars)  # check validation of configurations
        self._resetNetwork()
        self.network.load_state_dict(pars)
        self.trained = True

    def _trainBatch(self, batch_data, batch_reward):
        '''
        Train the network with a batch of data.
        :param batch_data: A batch of trials. 
        :param batch_reward: The reward of this batch of data.
        :return: 
            - total_loss.item(): The loss of this batch.
            - correct_rate(float): Correct rate of predictions of this batch of data.
        '''
        # TODO: set default value of block_reward
        self.network.zero_grad()
        # Initialization
        total_loss = 0
        times_num = batch_data.shape[0] - 1 # the number of time steps in this batch
        save_step = int(batch_data.shape[0]/2-1)
        predicted_trial = np.zeros((times_num, self.network.batch_size,self.network.in_dim))
        raw_prediction = np.zeros((times_num, self.network.batch_size, self.network.in_dim))
        hidden_sequence = np.zeros((times_num, self.network.batch_size, self.network.hid_dim))
        reward_guide = np.tile(batch_reward[:, 0], (batch_data.shape[2], 1))
        reward_guide = np.tile(reward_guide, (batch_data.shape[0], 1, 1)).transpose(0, 2, 1)
        # determine how many trials are used for training
        for i in range(self.network.batch_size):
            reward_guide[batch_reward[i, 1]:, i, :] = 0
        reward_guide = torch.tensor(reward_guide, dtype = torch.float, requires_grad=True)##########################
        reward_guide = reward_guide.cuda() if self.cuda_enabled else reward_guide
        # Train the network with each trial
        hidden = torch.tensor(self.hidden, dtype=torch.float32, requires_grad=True)
        for i in range(times_num):
            cur_time = np2tensor(batch_data[i:i+1, :, :], cuda_enabled = self.cuda_enabled) # current time step in the batch
            time_reward = reward_guide[i]
            next_time = np2tensor(batch_data[i+1:i+2, :, :], gradient_required=True) # next time step; used for computing loss
            # Train with current time step
            cur_time = torch.tensor(cur_time, dtype=torch.float32, requires_grad=True)
            output, hidden = self.network(cur_time, hidden)
            if i == save_step:  # TODO: not elegant, it is only for two step task
                self.hidden = copy.deepcopy(torch.tensor(hidden, dtype=torch.float32, requires_grad=True))
            # Compute  loss
            time_reward = torch.tensor(time_reward, dtype=torch.double, requires_grad=True)
            output = torch.tensor(output, dtype=torch.double, requires_grad=True)
            loss = self.network.criterion((time_reward * output).reshape([1, -1])[0],
                                          (time_reward * next_time).reshape([1, -1])[0])
            raw_output = tensor2np(output, cuda_enabled=self.cuda_enabled)
            copied_hidden = tensor2np(hidden, cuda_enabled=self.cuda_enabled)
            total_loss = total_loss + loss
            # Collect training results for each trial
            predicted_trial[i, :, :] = np.around(raw_output) # prediction is 1 if raw_output >= 0.5, else 0
            raw_prediction[i, :, :] = raw_output
            hidden_sequence[i, :, :] = copied_hidden
        # Backward passing
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        # Gradient descent
        self.network.optimizer.step()
        # Compute correction rate: the prediction[0] is estimation of trial[0], hence compare with trial[1]
        correct_rate = match_rate(batch_data[1:, :, :], predicted_trial)
        return total_loss.item(), correct_rate

    def _trainTimeStep(self, cur_time, time_reward, next_time):
        '''
        Train the network with a single trial.
        :param cur_time: Current time step.
        :param trial_reward: Reward of this time step.
        :param next_time: Next time step.
        :return: 
            - loss(float): Loss of prediction of this trial.
            - raw_output: Raw predicted output of this trial.
            - copied_hidden: Current hidden units value after training with this trial.
        '''
        cur_time = torch.tensor(cur_time, dtype=torch.float32,requires_grad=True)
        self.hidden=torch.tensor(self.hidden, dtype=torch.float32,requires_grad=True)
        output, hidden = self.network(cur_time, self.hidden)
        next_time = torch.tensor(next_time, dtype=torch.double,requires_grad=True)
        time_reward=torch.tensor(time_reward, dtype=torch.double,requires_grad=True)
        output=torch.tensor(output, dtype=torch.double,requires_grad=True)
        loss = self.network.criterion((time_reward * output).reshape([1, -1])[0],
                         (time_reward * next_time).reshape([1, -1])[0])
        raw_output = tensor2np(output, cuda_enabled = self.cuda_enabled)
        copied_hidden =tensor2np(self.hidden, cuda_enabled = self.cuda_enabled)
        return loss, raw_output, copied_hidden

    def _initHidden(self):
        '''
        Initialize the hidden units.
        :return: 
            - new_hidden (torch.tensor): Initialized hidden units.
        '''
        weight = next(self.network.parameters()).data
        # for pertubation
        noise = torch.randn(self.network.nlayers, self.network.batch_size, self.network.hid_dim,requires_grad=True) * self.init_noise_amp
        new_hidden = (weight.new(self.network.nlayers, self.network.batch_size, self.network.hid_dim).zero_() + noise)\
            .clone().requires_grad_(True)
        if self.cuda_enabled:
            return new_hidden.cuda()
        else:
            return new_hidden

    def _resetNetwork(self):
        '''
        Reinitialized the GRU network configurations. 
        Precondition: A feasible configuration dict is already loaded as self.config_pars. 
        :return: VOID
        '''
        self.network = _GRUNetwork(self.config_pars['in_dim'], self.config_pars['batch_size'])
        self.cuda_enabled = self.config_pars['cuda_enabled']
        self.init_noise_amp = self.config_pars['init_noise_amp']
        self.reset_hidden = self.config_pars['reset_hidden']
        self.gradient_clip = self.config_pars['gradient_clip']
        # TODO: determine whether need a log or not

########################################################################################################################
########################################################################################################################

class _GRUNetwork(nn.Module):
    '''
    Description: 
        A neural network class has three layers: one input layer, one hidden layer (with 128 GRUs), and one output layer. 
    
    Functions:
        __init__: Initialize the GRU network.
        forward: Forward passing function.
    
    Variables:
        in_dim: Dimension of input layer.
        out_dim: Dimension of output layer, fixed to be equal to in_dim in this network.
        hid_dim: The dimension of hidden layer, i.e., the number of GRUs in hidden layer, fixed to 128 in this network.
        nlayers: The number of hidden layers, fixed to 1 in this network.
        lr: Learning rate.
        cuda_enabled: Whether CUDA is used. 
        rnn: The RNN network provided by Pytorch.
        decoder: Mapping function between hidden layer and output layer, fixed to be linear function in this network.
        optimizer: Optimizing function for minimizing network loss. 
        criterion: Network loss function, fixed to be MSE loss in this network.
    '''

    def __init__(self, in_dim, batch_size=1, lr=1e-4, optimizer = "Adam", cuda_enabled=False):
        '''
        Initialize the neural network with three layers.
        :param in_dim: The size of one input sample, should be a positive integer.
        :param batch_size: The number of batches (1 by default), should be a positive integer.
        :param lr: Learning rate (1e-4 by default), should be a positive float number.
        :param init_noise_amp: Coefficient of amplifying noise when initializing network variables (0.0 by default), 
                                should be a positive float number. 
        :param optimizer: Optimizer for minimizing the network loss, either 'Adam' or 'SGD' ('Adam' by default).
        :param cuda_enabled: Whether CUDA is used, either True or False (False by default).
        '''
        super(_GRUNetwork, self).__init__()
        # Initialize network parameters
        self.in_dim = in_dim
        self.out_dim  = in_dim
        self.hid_dim = 128
        self.nlayers = 1
        self.batch_size = batch_size
        #self.bg_noise_amp = bg_noise_amp
        self.cuda_enabled = cuda_enabled
        self.lr = lr
        # Initialize hidden layers and mapping functions between layers
        self.rnn = nn.GRU(self.in_dim, self.hid_dim, num_layers=self.nlayers) # hidden layers with GRU
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(self.hid_dim, self.out_dim) # linear mapping between hidden layer and output layer
        # Optimizer should be Adam or SGD
        if optimizer is 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer is 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError('Unsupported optimizer type!')
        # Loss function of network
        self.criterion = nn.MSELoss(reduction='mean')


    def forward(self, input, hidden):
        '''
        Overloaded forward passing function fot Pytorch.nn.Module.
        :param input: Input data with shape of (seq_len, batch, input_size)
        :param hidden: Hidden unit value.
        :return: Output and hidden value of this forward passing:
            - output: Output value at this passing.
            - hidden: Hidden unit value at this passing.
        '''
        output, hidden = self.rnn(input, hidden)
        hidden = self.relu(hidden)
        output = self.relu(output)
        output = output.view(self.batch_size, -1)
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output, hidden

########################################################################################################################
########################################################################################################################

if __name__ == '__main__':
    network = TorchNetwork("test_config.json")
    rnn=_GRUNetwork(10)
    print(rnn)