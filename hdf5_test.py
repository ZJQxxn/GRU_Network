import h5py

logFileName = "RewardAffectData-OldTraining-OldNetwork-Three-Armed-Bandit/RewardAffectData-OldTraining-OldNetwork-ThreeArmed-sudden-reverse-model1-validation-1e6.hdf5"
logFile = h5py.File(logFileName, 'r')

choices = logFile['choice'].value
rewards = logFile['reward'].value
logFile.close()

print()



