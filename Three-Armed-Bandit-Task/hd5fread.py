import h5py
from scipy.io import loadmat

# with h5py.File('validate_record-three-armed-2019_11_29.hdf5', 'r') as file:
#     key = file.keys()
#     print()

mat = loadmat('./data/ThreeArmedBandit_TrainingSet-reverse-2019_11_29-1.mat')
print()