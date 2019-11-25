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
    '''
    def __init__(self):
        super(Task, self).__init__()