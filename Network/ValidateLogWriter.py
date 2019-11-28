'''
ValidateLogWriter.py: The abstrac class for writing log file for validating.

Author: Jiaqi Zhang
Date: Nov. 27 2019
'''

from abc import ABC, abstractmethod

class ValidateLogWriter(ABC):
    '''
    Description:
        Write hdf5 log file for validating.        
    
    Functions:
        __init__: Initialization.
        craeteHdf5File: Create log file.
        appendRecord: Append a validating record into log file.
        closeHdf5File: Close log file.
    '''

    def __init__(self, filename = ''):
        '''
        Initialization.
        :param filename: Log file name, should be a hdf5 file. 
        '''
        super(ValidateLogWriter, self).__init__()

    @abstractmethod
    def craeteHdf5File(self, data_shape):
        '''
        Create log file.
        :param data_shape: Shape of data for creating the hdf5 file.
        :return: VOID
        '''
        pass

    @abstractmethod
    def appendRecord(self, record):
        '''
        Append a validating record into log file.
        :param record: A validating record.
        :return: VOID
        '''
        pass

    @abstractmethod
    def closeHdf5File(self):
        '''
        Close log file.
        :return: VOID
        '''
        pass