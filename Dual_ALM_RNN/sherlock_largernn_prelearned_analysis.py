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


exp.configs['xs_left_alm_amp'] = 1.0
exp.configs['xs_right_alm_amp'] = 1.0
# exp.configs['random_seed'] = seed
exp.configs['train_type'] = 'train_type_modular_asymmetric_fixed_input'
exp.configs['n_neurons'] = 128
# exp.configs['unfix_epoch'] = epoch
exp.configs['bs'] = 256
exp.configs['lr'] = 1e-4
exp.configs['n_epochs'] = 30

os.makedirs('results/largernn_prelearned', exist_ok=True)

for unfix_epoch in [1, 10, 20, 30]:

    os.makedirs('results/largernn_prelearned/epoch_{}'.format(unfix_epoch), exist_ok=True)

    all_readout_acc_right = []

    for seed in range(100):
        # Load results_dict for each unfix_epoch
        results_dict = np.load(
            '/home/users/cwang314/dual_alm_rnn_logs/{}/{}/n_neurons_128_random_seed_{}/unfix_epoch_{}/n_epochs_30_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(
                exp.configs['model_type'],
                exp.configs['train_type'],
                seed,
                unfix_epoch,
                float(exp.configs['xs_left_alm_amp']),
                float(exp.configs['xs_right_alm_amp'])
            ),
            allow_pickle=True
        )

        readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])
        all_readout_acc_right.append(readout_acc_right)


    np.save('results/largernn_prelearned/epoch_{}/all_readout_acc_right.npy'.format(unfix_epoch), all_readout_acc_right)