import h5py
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

with h5py.File('validate_record-three-armed-2019_12_08-sudden_reverse.hdf5', 'r') as file:
    type = 'sudden_reverse'
    key = file.keys()
    prediction = file['prediction']
    choice = file['choice']
    index = 0
    trial_count = 0
    large_blk_size = 300
    block_num = len(choice)//large_blk_size
    count = np.zeros((block_num, large_blk_size))
    try:
        for step, trial in enumerate(choice):
            blk_index = step // large_blk_size
            count[blk_index, step % large_blk_size] = trial.item()
    except:
        print()
    count = count.astype(int)
    I = np.ones((block_num, 150))
    c = np.hstack((0*I, 0*I)) if type is 'fixed' else np.hstack((0*I, 2*I))
    match = np.array(count == c).astype(int)
    prob = np.mean(match, axis = 0)
    plt.plot(np.arange(0, large_blk_size, 1), prob)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xlabel('trial number')
    plt.ylabel('correct rate')
    plt.show()
    # choice = file['choice']
    # reward = file['reward']
    # # correct_rate = file['correct_rate']
    # reward_count = [0, 0]
    # choice_count = [0, 0, 0, 0]
    # trial_count = [0, 0]
    # total_choice = []
    # # for index in range(51, 100):
    # for index in range(choice.shape[0] // 50):
    #     start = index * 50
    #     for i in range(start, start+50):
    #         if choice[i] < 2:
    #             if choice[i] == 0:
    #                 trial_count[0] += 1
    #             else:
    #                 trial_count[1] += 1
    #     total_choice.append(trial_count)
    #     trial_count = [0, 0]



