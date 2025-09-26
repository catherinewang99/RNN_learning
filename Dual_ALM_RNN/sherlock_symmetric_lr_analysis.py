import os
import sys
import json
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp
import matplotlib.pyplot as plt
from dual_alm_rnn_models import *
import argparse, os, math, pickle, json
plt.rcParams['pdf.fonttype'] = '42' 

exp = DualALMRNNExp()

# Load configs to get model parameters
with open('dual_alm_rnn_configs.json', 'r') as f:
    configs = json.load(f)


# for unfix_epoch in [1, 11, 40]:
unfix_epoch = 40
os.makedirs('figs/lr', exist_ok=True)

# left_lr = []
# all_learning_right = []
# all_learning_left = []

# for seed in range(100):

#     # get learning epoch
#     logs_path = '/home/users/cwang314/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_fixed_input_cross_hemi_mult_seeds/n_neurons_4_random_seed_{}/unfix_epoch_{}/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_1.00/init_cross_hemi_rel_factor_0.20'
#     results_path = os.path.join(logs_path.format(seed, unfix_epoch), 'all_val_results_dict.npy')
#     if not os.path.exists(results_path):
#         continue
#     results_dict = np.load(results_path, allow_pickle=True)
#     readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])
#     readout_acc_left = np.array([results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))])

#     all_learning_right.append(readout_acc_right)
#     all_learning_left.append(readout_acc_left)




# np.save('figs/lr/all_learning_right.npy', all_learning_right)
# np.save('figs/lr/all_learning_left.npy', all_learning_left)


for asymm in [(1, 0.2), (0.2, 1)]:

    all_learning_right = []
    all_learning_left = []

    for seed in range(100):

        # get learning epoch
        logs_path = '/home/users/cwang314/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_fixed_input_cross_hemi_mult_seeds/n_neurons_4_random_seed_{}/unfix_epoch_{}/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20'
        results_path = os.path.join(logs_path.format(seed, unfix_epoch, asymm[0], asymm[1]), 'all_val_results_dict.npy')
        if not os.path.exists(results_path):
            continue
        results_dict = np.load(results_path, allow_pickle=True)
        readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])
        readout_acc_left = np.array([results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))])

        all_learning_right.append(readout_acc_right)
        all_learning_left.append(readout_acc_left)




    np.save('figs/lr/all_learning_right_L{}_R{}.npy'.format(asymm[0], asymm[1]), all_learning_right)
    np.save('figs/lr/all_learning_left_L{}_R{}.npy'.format(asymm[0], asymm[1]), all_learning_left)