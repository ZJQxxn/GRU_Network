import h5py
from scipy.io import loadmat
with h5py.File('validation-2019_12_06.hdf5', 'r') as file:
    key = file.keys()
    choice = file['choice']
    reward = file['reward']
    reward_count = [0, 0]
    choice_count = [0, 0, 0, 0]
    # for index in range(51, 100):
    for index in range(choice.shape[0]):
        if reward[index][0] < 2:
            reward_count[reward[index][0]] += 1
        if choice[index][0] < 4:
            choice_count[choice[index][0]] += 1

    print()


