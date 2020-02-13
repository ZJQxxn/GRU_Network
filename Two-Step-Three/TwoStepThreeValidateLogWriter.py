'''
TwoStepValidateLogWriter.py: Write log file for validating of the two-step task. 

Author: Jiaqi Zhang <zjqseu@gmail.com>
Date: Nov. 27 2019
'''

import sys
sys.path.append('../Network')

from ValidateLogWriter import ValidateLogWriter

import tables
import numpy as np


class TwoStepThreeValidateLogWriter(ValidateLogWriter):
    '''
    Description:
        Write log file for validating.
        
    Variables:
        low: Line numuber for writing into the hdf5 file.
         filename: Hdf5 log file name.
         hdf5file: The log file handle.
    
    Functions:
        __init__: Initalization.
        craeteHdf5File: Create the log file.
        appendRecord: Append a validating record.
        closeHdf5File: Close the log file.
    '''

    def __init__(self, filename):
        '''
        Initialization.
        :param filename: Log file name.
        '''
        self.low = 0
        self.filename = filename

    def craeteHdf5File(self, data_shape):
        '''
        Create the hdf5 log file.
        :param data_shape: A dict of data shape used for creating the hdf5 file.
        :return: VOID
        '''
        self.hdf5file = tables.open_file(self.filename, mode='w')
        content_type = tables.Float64Atom()
        index_dtype = tables.UInt64Atom()
        neuron_shape = data_shape['neuron_shape']
        behavior_shape = data_shape['behavior_shape']
        label_shape = (0, 1, 2)
        neuron_shape.insert(0, 0)
        behavior_shape.insert(0, 0)
        self.behavior_storage = self.hdf5file.create_earray(self.hdf5file.root, 'behavior', content_type,
                                                                shape=behavior_shape)
        self.prediction_storage = self.hdf5file.create_earray(self.hdf5file.root, 'prediction', content_type,
                                                                  shape=behavior_shape)
        self.rawoutput_storage = self.hdf5file.create_earray(self.hdf5file.root, 'rawoutput', content_type,
                                                                 shape=behavior_shape)
        self.neuron_storage = self.hdf5file.create_earray(self.hdf5file.root, 'neuron', content_type,
                                                              shape=neuron_shape)
        self.index_storage = self.hdf5file.create_earray(self.hdf5file.root, 'index', index_dtype,
                                                             shape=label_shape)
        self.choice_storage = self.hdf5file.create_earray(self.hdf5file.root, 'choice', index_dtype,
                                                          shape=(0, 1))
        self.reward_storage = self.hdf5file.create_earray(self.hdf5file.root, 'reward', index_dtype,
                                                          shape=(0, 1))
        self.correct_rate_storage = self.hdf5file.create_earray(self.hdf5file.root, 'correct_rate', content_type,
                                                          shape=(0, 1))

    def appendRecord(self, record):
        '''
        Append a validating record.
        :param record: A dict of two-step tasks validating record.
        :return: VOID
        '''
        # Extract record
        behavior_data = np.array(record["sensory_sequence"])
        prediction_data = np.array(record["predicted_trial"])
        raw_output = np.array(record["raw_records"])
        neuron_data = np.array(record["hidden_records"])
        tmp_high = self.low + behavior_data.shape[0]
        index_data = np.array([self.low, tmp_high]).reshape(1, 1, 2)
        # Append record into log file
        self.behavior_storage.append(behavior_data)
        self.prediction_storage.append(prediction_data)
        self.rawoutput_storage.append(raw_output)
        self.neuron_storage.append(neuron_data)
        self.index_storage.append(index_data)
        self.choice_storage.append(np.array([record['choice']]).reshape(1, 1))
        self.reward_storage.append(np.array([record['reward']]).reshape(1, 1))
        self.correct_rate_storage.append(np.array([record['correct_rate']]).reshape(1, 1))
        self.low = tmp_high

    def closeHdf5File(self):
        '''
        Close the log file handle.
        :return: VOID
        '''
        self.hdf5file.close()