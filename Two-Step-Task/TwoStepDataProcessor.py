'''
TwoStepDataProcessor.py: Data processor for the two-step task.

Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Nov. 27 2019
'''

from Network.DataProcessor import DataProcessor
from scipy.io import loadmat

class TwoStepDataProcessor(DataProcessor):
    '''
    Decription: 
        Data processor for the two-step task.
        
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
        super(TwoStepDataProcessor, self).__init__()
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
        self.train_data_set = [t[0] for t in mat['data_ST']]
        self.train_guide = mat['data_ST_Brief']['training_guide'][0][0]
        return self.train_data_set, self.train_guide

    def prepareValidatingData(self):
        '''
        Prepare validating data.
        :return: 
            validate_data_set: Validating dataset.
            validate_data_attr: Validating data attributes, including reward probability, etc.
        '''
        mat = loadmat(self.validate_data_file)
        self.validate_data_set = [t[0] for t in mat['data_ST']]
        self.validate_data_attr = mat['data_ST_Brief']
        return self.validate_data_set, self.validate_data_attr


if __name__ == '__main__':
    p = TwoStepDataProcessor('./data/SimpTwo_TrainingSet-2019_11_19-1.mat',
                             './data/SimpTwo_TestingSet-2019_11_19-1.mat',
                            )
    train_set, train_guide = p.prepareTrainingData()
    validate_set, validate_attr = p.prepareValidatingData()
    print("The size of a training trial is:", train_set[0].shape)
    print("The size of a train guide is:", train_guide.shape)
    print("The size of a validating trial is:", validate_set[0].shape)
