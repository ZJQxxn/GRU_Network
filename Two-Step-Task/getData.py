#TODO: test and revise this
from Network.net_tools import readConfigures, validateConfigures
import os
import scipy
from Network.DataProcessor import DataProcessor
import time


class GetData(DataProcessor):
    def __init__(self, config_file=""):
        '''
        Initialization.
        '''
        self.config_pars = readConfigures(config_file)
        self.config_pars = validateConfigures(self.config_pars)

    def load_data_v2(self, data_path, data_attr, data_brief_attr):
        """
        Turn data into the numpy format.
        """
        print("Current Path: ", os.getcwd())
        self.data_path = data_path
        self.data_attr = data_attr
        mat = scipy.io.loadmat(data_path)
        #        self.data_set = np.array([np.transpose(t[0]).astype(np.int) for t in mat[data_attr]])
        #        self.data_set = [np.transpose(t[0]) for t in mat[data_attr]]
        self.data_set = [t[0] for t in mat[data_attr]]
        self.data_conditions = mat[data_brief_attr][0, 0]
        return self.data_set, self.data_conditions

    def prepareTrainingData(self):
        training_data_path = self.config_pars['data_file']
        data_attr = self.config_pars['data_attr']
        data_brief_attr = self.config_pars['data_brief']
        training_set, training_conditions = self.load_data_v2(training_data_path, data_attr, data_brief_attr)
        training_data = {
            "training_set": training_set,
            "training_conditions": training_conditions
        }
        return training_data

    def prepareTestingData(self):
        validation_data_path = self.config_pars['validation_data_file']
        data_attr = self.config_pars['data_attr']
        data_brief_attr = self.config_pars['data_brief']
        validation_set, validation_conditions = self.load_data_v2(
            validation_data_path, data_attr, data_brief_attr)
        validation_data = {
            "validation_set": validation_set,
            "validation_conditions": validation_conditions
        }
        return validation_data

