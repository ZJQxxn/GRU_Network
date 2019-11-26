'''
Task.py: Abstract class for tasks.

Author: Jiaqi Zhang <zjqseu@gmail.com.
Date: Nov. 16 2019
'''

from abc import ABC, abstractmethod

class Task(ABC):
    '''
    Abstract class for different experiment tasks.
    
    Functions:
        __init__: Initialization.
        train: Training on this task.
        validate: Validating on this task.
    '''
    def __init__(self, config_file = ""):
        super(Task, self).__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass