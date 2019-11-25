'''
net_tool.py: Utility functions.

Author: Jiaqi Zhang <zjqseu@gmail.com.
Date: Nov. 15 2019
'''

import json
import numpy as np
import torch

def readConfigures(filename):
    '''
    Read in a json configuration file for the neural network.
    :param filename: The filename of configuration settings.
    :return: A dict of all the configuration parameters.
    '''
    with open(filename, 'r') as file:
        config_pars = json.load(file)
    return config_pars

def validateConfigures(config_pars):
    # check require configuration settings
    if 'in_dim' not in config_pars:
        raise ValueError('Miss in_dim as the dimensionality of input layer.')
    # check optional configuration settings, if not provided, set to the default value
    if 'cuda_enabled' not in config_pars:
        config_pars['cuda_enabled'] = False
    #TODO: set optional parameters by their default values if they are not provided in the configuration file
    return config_pars

def tensor2np(torch_tensor, cuda_enabled = False):
    '''
    Convert a Pytorch tensor to Numpy ndarray.
    :param torch_tensor: Pytorch tensor.
    :param cuda_enabled: Whether CUDA is available, either True or False (set to False by default).
    :return: A Numpy ndarray.
    '''
    if cuda_enabled:
        return torch_tensor.data.cpu().numpy()  # pull data from GPU to CPU
    else:
        return torch_tensor.data.numpy()

def np2tensor(nparray, block_size = 1, cuda_enabled = False, gradient_required = True):
    '''
    Convert a Numpy ndarray to a Pytorch tensor.
    :param nparray: A numpy ndarray.
    :param block_size: Block size, should be a positive integer (set to 1 by default).
    :param cuda_enabled: Whether CUDA is available, either True or False (set to False by default).
    :param gradient_required: Whether automatic gradient computing is required, either True of False (set to True by default).
    :return: A Pytorch tensor with shape of (1, block_size, -1).
    '''
    if cuda_enabled:
        return torch.tensor(torch.tensor(nparray).view(1, block_size, -1).cuda(), requires_grad=gradient_required)
    else:
        return torch.tensor(torch.tensor(nparray).view(1, block_size, -1), requires_grad=gradient_required)

def match_rate(t1, t2):
    '''
    Compute the matching rate between two tensors.
    :param t1: The first tensor.
    :param t2: The second tensor.
    :return: The matching rate between two tensors. If the shape of two tensors mismatch, return 0.
    '''
    if t1.shape == t2.shape:
        diff = np.sum(np.abs(t1 - t2))  # trial is binary
        return 1 - (diff / (t1.size))
    else:
        print("Matched shape? No!")
        return 0

if __name__ == '__main__':
    config_pars = readConfigures('test_config.json')
    config_pars = validateConfigures(config_pars)
    print(config_pars)