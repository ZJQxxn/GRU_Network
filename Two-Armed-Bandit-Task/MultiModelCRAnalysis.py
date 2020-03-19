'''
Description: 
    Compute the correct rate of the vaslidation result.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    10 Feb. 2020
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt


blk_size = 70

def crAnalyze(filename):
    with h5py.File(filename, 'r') as f:
        choices = np.array(f['choice'].value, dtype=np.float32)

    # the number of trials
    numTrials = len(choices)
    numBlks = numTrials // blk_size if (numTrials // blk_size) % 2 == 0 \
        else (numTrials // blk_size) - 1  # number of small blocks
    numTrials = numBlks * blk_size
    choices = choices[:numTrials]

    # A1 is reward in the first block; A2 is reward in the second block
    block_indication = np.hstack(
        (np.zeros(blk_size, ), np.ones(blk_size, ))
    )
    block_indication = np.tile(block_indication, numBlks // 2)

    # find out where the block reverse
    block_sep = np.where(np.diff(block_indication) != 0)[0]
    numWholeBlocks = len(block_sep)

    # correct rate
    correct_rate = []
    for whole_blk_index in range(numWholeBlocks - 1):
        reverse_point = block_sep[whole_blk_index]
        # next_reverse_points = block_sep[whole_blk_index+1]
        cr_range = np.arange(reverse_point - 10, reverse_point + 30)
        cr = [
            1 if choices[trial] == block_indication[trial]  # choice is 1/2, indication is 0/1
            else 0
            for trial in cr_range
        ]
        correct_rate.append(cr)
    correct_rate = np.vstack(correct_rate)
    correct_rate = np.nanmean(correct_rate, axis=0)
    return correct_rate


if __name__ == '__main__':
    model_cr = np.zeros((5, 8, 40))  # 5 models, 8 intermediate models, 40 trials around th reversla points is analyzed
    for num in range(1, 6):
        print('============ MODEL {} ==========='.format(num))
        # Validate intermediate model
        for index in range(1, 8):
            print('------- intermediate model {}'.format(index))
            filename = 'save_m/model_{}/TwoArmed-validation-75e5-model{}-NUM{}.hdf5'.format(num, num, index)
            correct_rate = crAnalyze(filename)
            model_cr[num-1, index-1, :] = correct_rate
        # Validate the final model
        correct_rate = crAnalyze('TwoArmed-validation-75e5-model{}.hdf5'.format(num))
        model_cr[num-1, -1, :] = correct_rate

    for inter_num in range(8):
        current_cr = model_cr[:, inter_num, :]
        mean_cr = np.mean(current_cr, axis = 0)
        var_cr = np.var(current_cr, axis = 0)
        # Plot result
        plt.figure(figsize=(18, 9))
        plt.errorbar(x = np.arange(len(mean_cr)), y = mean_cr, yerr=var_cr, capsize=5)
        plt.title('Correct Rate vs. Trial  [trialNum = {}e5]'.format(inter_num+1), fontsize=20)
        plt.xlabel('lag (trials)', fontsize=20)
        plt.ylabel('correct rate', fontsize=20)
        plt.yticks(np.linspace(0.0, 1.0, num=11, endpoint=True), fontsize=20)
        plt.xticks(np.arange(40), np.arange(0, 41) - 10, fontsize=10)
        plt.savefig('cr/average_cr/TwoArmed-AverageCR-{}e5.pdf'.format(inter_num+1))
        # plt.show()



