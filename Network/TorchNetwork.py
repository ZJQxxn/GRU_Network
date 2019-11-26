'''
TorchNetwork.py: Implementation of the class for our gated neural unit (GRU) network with Pytorch.

Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Nov. 15 2019

Reference: 

'''
import time
import torch
import torch.nn as nn
import numpy as np

from Network.net_tools import  readConfigures, validateConfigures
from Network.net_tools import tensor2np, np2tensor, match_rate


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
        Initialize the GRU network. If no configuration file is provided, the model will be loaded through a .pt file.
        :param config_file: Name of the configuration file, should be a JSON file name of srting type ("" by default).
        '''
        if config_file is not "":
            self.config_pars = readConfigures(config_file) # read configurations from file
            self.config_pars = validateConfigures(self.config_pars) # check validation of configurations
            self._resetNetwork() # initialize the network with a configuration file
        self.trained = False # denote whether the network has been trained

    def training(self, train_set, training_guide, truncate_iter = 1e800):
        '''
        Train the GRU  neural network.
        :param train_set: Training dataset.
        :param training_guide: Training reward.
        :param truncate_iter: Truncated iteration. The network training stops when this number of trials are trained.
        :return: 
            - train_loss(list0: Losses of every block.
            - train_correct_rate(list): Correct rates of every block.
        '''
        self.hidden = self._initHidden()
        print("self.hidden:",self.hidden);
        block_data = []
        block_reward = []
        train_loss = []
        train_correct_rate = []
        for step, trial in enumerate(train_set):
            # cease training when reaching at the truncate iteration
            if step >= (truncate_iter - 1):
                break
            # Collect ``batch_size'' number of trials as training input
            block_data.append(trial)
            block_reward.append(training_guide[step])
            # training for every batch
            if (step + 1) % self.network.batch_size == 0:
                # reset hidden unit for next batch
                if self.reset_hidden:
                    self.hidden = self._initHidden()
                # train the network with current block of data
                block_data = np.array(block_data).transpose([1, 0, 2])
                #加入block_reward=np.array(block_reward)
                block_reward = np.array(block_reward)# train_input: (time step, batch size, input number)
                block_loss, block_correct_rate = self._trainBlock(block_data, block_reward)
                block_data,block_reward=[],[]
                train_loss.append(block_loss)
                train_correct_rate.append(block_correct_rate)
                # TODO: print out or write into log file?
                print("Network loss is : ", block_loss)
                print("Correct rate is : ", block_correct_rate)
                print('=' * 20)
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
        pars['configurations'] = self.config_pars
        torch.save(pars, filename)

    def loadModel(self, filename):
        '''
        Load Pytorch network from .pt file, with all the network training configurations.
        :param filename: Filename of .py file.
        :return: VOID
        '''
        #TODO: What if no configurations in the .pt file
        pars = torch.load(filename)
        self.config_pars = pars.pop('configurations')
        self._resetNetwork()
        self.network.load_state_dict(pars)
        self.trained = True

    def _trainBlock(self, block_data, block_reward):
        '''
        Train the network with a block of data.
        :param block_data: A block of trials. 
        :param block_reward: The reward of this block of data.
        :return: 
            - total_loss.item(): The loss of this block.
            - correct_rate(float): Correct rate of predictions of this block of data.
        '''
        self.network.zero_grad()
        # Initialization
        total_loss = 0
        block_size = block_data.shape[0] - 1 # in case data is truncated
        #
        predicted_trial = np.zeros((block_size, self.network.batch_size,self.network.in_dim))
        raw_prediction = np.zeros((block_size, self.network.batch_size, self.network.in_dim))
        hidden_sequence = np.zeros((block_size, self.network.batch_size, self.network.hid_dim))
        reward_guide = np.tile(block_reward[:, 0], (block_data.shape[2], 1))
        reward_guide = np.tile(reward_guide, (block_data.shape[0], 1, 1)).transpose(0, 2, 1)
        for i in range(self.network.batch_size):
            reward_guide[block_reward[i, 1]:, i, :] = 0
        reward_guide = torch.tensor(reward_guide, dtype = torch.float, requires_grad=True)##########################
        reward_guide = reward_guide.cuda() if self.cuda_enabled else reward_guide
        # Train the network with each trial
        for i in range(block_size):
            cur_trial = np2tensor(block_data[i, :, :], block_size = self.network.batch_size, cuda_enabled = self.cuda_enabled)
            trial_reward = reward_guide[i]
            n_put = np2tensor(block_data[i + 1, :, :], gradient_required=True) # TODO: The meaning of n_put
            loss, raw_output, copied_hidden = self._trainTrial(cur_trial, trial_reward, n_put)
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
        # Compute correct rate of training prediction
        # TODO: use a user defined match rate function. Necessary for every task actually?
        correct_rate = match_rate(block_data[1:, :, :], predicted_trial) #TODO: predicted_trial has shape (26, 1, 10)
        return total_loss.item(), correct_rate

    def _trainTrial(self, cur_trial, trial_reward, n_put):
        '''
        Train the network with a single trial.
        :param cur_trial: Current trial.
        :param trial_reward: Reward of this trial.
        :param n_put: TODO: the meaning of n_put.
        :return: 
            - loss(float): Loss of prediction of this trial.
            - raw_output: Raw predicted output of this trial.
            - copied_hidden: Current hidden units value after training with this trial.
        '''
        #下面两句为添加内容，将数据转为tensor.float32 #TODO: float32 or float64
        cur_trial = torch.tensor(cur_trial, dtype=torch.float32,requires_grad=True)
        self.hidden=torch.tensor(self.hidden, dtype=torch.float32,requires_grad=True)
        output, self.hidden = self.network(cur_trial, self.hidden)
        n_put = torch.tensor(n_put, dtype=torch.double,requires_grad=True)
        trial_reward=torch.tensor(trial_reward, dtype=torch.double,requires_grad=True) # TODO: the reward should be a parameter of network
        output=torch.tensor(output, dtype=torch.double,requires_grad=True)
        loss = self.network.criterion((trial_reward * output).reshape([1, -1])[0],
                         (trial_reward * n_put).reshape([1, -1])[0])
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

        #self.network.in_dim改为self.network.hid_dim
        noise = torch.randn(self.network.nlayers, self.network.batch_size, self.network.hid_dim,requires_grad=True) * self.init_noise_amp
        new_hidden = torch.tensor(
            weight.new(self.network.nlayers, self.network.batch_size, self.network.hid_dim).zero_() + noise
        ,requires_grad=True)
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
        self.network = _GRUNetwork(self.config_pars['in_dim'])
        self.cuda_enabled = self.config_pars['cuda_enabled']
        self.init_noise_amp = self.config_pars['init_noise_amp']
        self.reset_hidden = self.config_pars['reset_hidden']
        self.gradient_clip = self.config_pars['gradient_clip']

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
        self.batch_size = batch_size #TODO: why define here? Omit from this class
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
        # TODO: shape of input, shape of hidden
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