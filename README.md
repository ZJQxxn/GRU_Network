# GRU_Network

## Description

## Project Structure
    ├── README.md                        // decription and help
    |── LICENSE                          // license
    ├── Analyzer                         // analyzing experimental results
    │   └── Analyzer.py            
    ├── Network                          // codes for the GRU network
    │   ├── DataProcessor.py             // abstract class of data processors
    │   ├── Task.py                      // abstract class for tasks
    │   ├── TFNetwork.py                 // the GRU network model implemented by Tensorflow
    │   ├── TorchNetwork.py              // the GRU network model implemented by Pytorch
    │   ├── ValidateLogWriter.py         // abstract class for writing the log of validating
    │   └── net_tools.py                 // some utility functions
    └── Two-Step-Task                    // experiments for two-step task
        ├── data                         // task data
        ├── save_m                       // already trained GRU network model
        ├── test.hdf5                    // validating log file
        ├── test_config.json             // task configurations
        ├── TwoStepDataGenerator.py      // generate task data
        ├── TwoStepDataProcessor.py      // prepare training and validating data for task
        ├── TwoStepValidateLogWriter.py  // write log file for validating in task
        └── TeoStepTask.py               // main file of experiments of task
    
## TODO
  - [ ] Implement  three-armed bandit task
  - [ ] Implement the GRU network with Tensorflow
  
  
## Use GRU Network

### A general guide of how to use the network model

### Tutorial I: directly use GRU network
 
### Tutorial II: used in the two-step task

### Tutorial III: used in the three-armed bandit task