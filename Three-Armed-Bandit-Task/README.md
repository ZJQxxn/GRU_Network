# Three-Armed Bandit Task

## Generate Training Data
The number of features is 10 and the number of time steps is 14. Totally 900 trials are generated as training dataset.

## Generate Validation Data
The number of features is 10 and the number of time steps generated for validation data is 5 (including 2 for blank time
and 3 for showing three stimulus). 5100 trials are generated as validation dataset.

## Trainig and Validation
The training and validation procedure are implemented in **ThreeArmedTask.py**. A pre-trained model is already saved 
in the directory **save_m**.  

## Description of Data in Validation Log File
The HDF5 log file is recorded in the validation. Notice that 15 time steps are recorded for every trial, though one 
trial only lasts 14 seconds. The 15-th time step is recorded for simplicity of validation and is not contained in the 
evaluation of correct rate and loss. The log file contains five parts:
* **behavior**: has the shape (5100 * 15, 10). Each time step represents the network input. 
                  For example, index [0, :] denotes the 0-th time step of the first trial.
* **prediction**: has the shape (5100 * 15, 10). Each time steps represent the network output after binarified. 
                  For example, index [0, :] denotes the estimated 1-th time step of the first trial.
* **rawoutput**: has the shape (5100 * 15, 10). Each time step represents the network output. 
                  For example, index [0, :] denotes the estimated 1-th time step of the first trial.
* **neurons**: has the shape (5100 * 15, 1, 1, 128). Each time step represents the value of network hidden units. 
                  For example, index [0, :] denotes the hidden units' value after updated with the 0-th input of the 
                  first trial.
* **index**: has the shape (5100, 1, 2). Represent the index of a certain trial. 
                  For example, index [0, :] = [0, 15] denotes the time step index of the first trial, which is from 0-th 
                  second to 15-th second. Therefore, for every record, index [0:16, :] represent all the data relevant 
                  to the first trial. 