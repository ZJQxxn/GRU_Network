'''
DataProcessor.py: Abstract class for data preprocessing.

Author: Jiaqi Zhang <zjqseu@gmail.com.
Date: Nov. 16 2019
'''

from abc import ABC, abstractmethod

class DataProcessor(ABC):
    '''
    Abstract class for processing experimental data. Functions for preparing training and 
    testing datasets should be implemented.
    
    Functions:
        __init__: Initialization.
        prepareTrainingData: Prepare training data.
        prepareValidatingData: Prepare validating data.
    '''

    def __init__(self):
        '''
        Initialization.
        '''
        super(DataProcessor, self).__init__()

    @abstractmethod
    def prepareTrainingData(self):
        '''
        Preparing training dataset.
        '''
        pass

    @abstractmethod
    def prepareValidatingData(self):
        '''
        Prepare validating dataset.
        '''
        pass

