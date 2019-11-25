from Network.TorchNetwork import TorchNetwork
import os
import scipy
import importlib
import numpy as np
from  data_util import DataProcessor # TODO: do not use this
import time


def getTrainData(network):
    training_data_path = network.config_pars['data_file']
    data_attr = network.config_pars['data_attr']
    data_brief_attr = network.config_pars['data_brief']
    dp = DataProcessor()
    training_set, training_conditions = dp.load_data_v2(training_data_path, data_attr, data_brief_attr)
    training_data = {
        "training_set": training_set,
        "training_conditions": training_conditions
    }
    task = importlib.import_module('tasks.' + network.config_pars["name"]) # TODO: change this
    training_helper = task.TrainingHelper(training_set, training_conditions)
    training_guide = training_helper.training_guide
    return training_data,training_guide


if __name__ == '__main__':
    network = TorchNetwork("test_config.json")
    training_data,training_guide=getTrainData(network)
    network.training(training_data['training_set'],training_guide)
    print(network.network)