import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = '42' 
from dual_alm_rnn_models import *
import argparse, os, math, pickle, json

exp = DualALMRNNExp()

logs_path = '/home/users/cwang314/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_asymmetric_fixed_input_mult_seeds/n_neurons_4_random_seed_{}/unfix_epoch_{}/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_1.00/init_cross_hemi_rel_factor_0.20'

os.makedirs('figs', exist_ok=True)

# make a plot that shows the acc for each seed at each of the epochs

f = plt.figure()
for seed in range(100):
    epoch_colors = {1: 'blue', 11: 'orange', 20: 'green', 40: 'red'}
    # for epoch in [0, 10, 20, 40]:
    for epoch in [1, 11, 40]:
        results_path = os.path.join(logs_path.format(seed, epoch), 'all_val_results_dict.npy')
        if not os.path.exists(results_path):
            continue
        results_dict = np.load(results_path, allow_pickle=True)
        readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])

        plt.plot(readout_acc_right, color=epoch_colors[epoch], label=f'Seed {seed}, Epoch {epoch}')
plt.xlabel('Epoch')
plt.ylabel('Readout Accuracy')
plt.title('Readout Accuracy vs Epoch')
plt.savefig('figs/symmetric_rnn_learning_trajectory.pdf')


# make a plot that shows the diff in learning rate compared to control for each seed at each of the epochs
all_epochtrained_diff = []
all_epochtrained = []

for seed in range(100):

    epoch_colors = {0: 'blue', 10: 'orange', 20: 'green', 40: 'red'}
    epochtrained_diff = []
    epochtrained = []
    ctl_epoch = 40

    # for unfix_epoch in [40, 0, 10, 20]:
    for unfix_epoch in [40, 1, 11]:
        results_path = os.path.join(logs_path.format(seed, unfix_epoch), 'all_val_results_dict.npy')

        if not os.path.exists(results_path):
            continue

        results_dict = np.load(results_path, allow_pickle=True)
        readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])


        # Add epoch where right side reaches 70% performance
        if len(np.where(readout_acc_right >= 0.7)[0]) == 0 and unfix_epoch == 40: # skip case where the rnn never learns
            pass
        elif unfix_epoch == 40:
            ctl_epoch = np.where(readout_acc_right >= 0.7)[0][0]
        elif len(np.where(readout_acc_right >= 0.7)[0]) == 0: # this regime never learns
            epochtrained.append(40)
            epochtrained_diff.append(40 - ctl_epoch)
        else:
            epochtrained.append(np.where(readout_acc_right >= 0.7)[0][0])
            epochtrained_diff.append(np.where(readout_acc_right >= 0.7)[0][0] - ctl_epoch)

    all_epochtrained.append(epochtrained)
    all_epochtrained_diff.append(epochtrained_diff)


np.save('figs/all_epochtrained.npy', all_epochtrained)
np.save('figs/all_epochtrained_diff.npy', all_epochtrained_diff)

