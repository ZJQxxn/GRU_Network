'''
ThreeArmedDataProcessor.py: Data processor for the three-armed bandit task.

Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Nov. 29 2019
'''
import sys
sys.path.append('../Network/')
from DataProcessor import DataProcessor
from scipy.io import loadmat

class ThreeArmedDataProcessor(DataProcessor):
    '''
    Decription: 
        Data processor for the three-armed bandit task.
        
    Variables:
        train_data_file: File name of training dataset.
        validate_data_file: File name of validating dataset.
        train_data_set: Training dataset.
        train_guide: Weight coefficient for training data.
        validate_data_set: Validating dataset.
        validate_data_attr: Validating data attributes, including reward probability, etc.
        
    Functions:
        __init__: Initialization.
        prepareTrainingData: Prepare training data.
        prepareValidatingData: Prepare validating data.
    '''

    def __init__(self, train_data_file, validate_data_file):
        '''
        Initialization.
        :param train_data_file: Training data filename. 
        :param validate_data_file: Validating data filename.
        '''
        super(ThreeArmedDataProcessor, self).__init__()
        self.train_data_file = train_data_file
        self.validate_data_file = validate_data_file


    def prepareTrainingData(self):
        '''
        Prepare training data.
        :return: 
            train_data_set: Training dataset.
            train_guide: Weight coefficient for training data.
        '''
        mat = loadmat(self.train_data_file)
        self.train_data_set = mat['training_set']
        self.train_guide = mat['training_guide']
        return self.train_data_set, self.train_guide

    def prepareValidatingData(self):
        '''
        Prepare validating data.
        :return: 
            validate_data_set: Validating dataset.
            validate_data_attr: Validating data attributes, including reward probability, etc.
        '''
        mat = loadmat(self.validate_data_file)
        self.validate_data_set = mat['validating_set']
        self.validate_data_attr = mat['info']
        return self.validate_data_set, self.validate_data_attr


if __name__ == '__main__':
    p = ThreeArmedDataProcessor('./data/ThreeArmedBandit_TrainingSet-reverse-2019_11_29-1.mat',
                             './data/ThreeArmedBandit_TestingSet-reverse-2019_11_29-1.mat',
                            )
    train_set, train_guide = p.prepareTrainingData()
    validate_set, validate_attr = p.prepareValidatingData()
    print("The size of a training trial is:", train_set[0].shape)
    print("The size of a train guide is:", train_guide.shape)
    print("The size of a validating trial is:", validate_set[0].shape)
