"""
Test the performance under perturbation conditions for various asymmetries in the model
"""

import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp
from test_perturbation_eval import test_perturbation_evaluation
import matplotlib.pyplot as plt


input_asym = [(1,0), (1,0.1), (1,0.2), (1,0.5), (1,1), (0.5,1), (0.2,1), (0.1,1), (0,1)] # Sane as BK

input_asym = [(0, 1)] # Sane as BK

cd_acc_left, cd_acc_right = [], []
cd_acc_rightpert_left, cd_acc_rightpert_right = [], []
control_acc_left, control_acc_right = [], []

# Initialize the experiment
exp = DualALMRNNExp()
for input_asym in input_asym:
    for seed in range(5,10):
        exp.configs['xs_left_alm_amp'] = input_asym[0]
        exp.configs['xs_right_alm_amp'] = input_asym[1]
        exp.configs['random_seed'] = seed
        exp.init_sub_path(exp.configs['train_type'])
        model_save_path = os.path.join(exp.configs['models_dir'], exp.configs['model_type'], exp.sub_path)
        model_path = os.path.join(model_save_path, 'best_model.pth')



        if os.path.exists(model_path):
            print(f"Model already exists at {model_path}, skipping training.")
        else:
            print(f"Training model ...")
            exp.train_type_modular()



#PLOT

# Plot control vs right/left pert accuracy
control_acc = []
right_pert_acc = []
left_pert_acc = []

control_std = []
right_pert_std = []
left_pert_std = []

input_asym = [(1.0,0.0), (1.0,0.1), (1.0,0.2), (1.0,0.5), (1.0,1.0), (0.5,1.0), (0.2,1.0), (0.1,1.0), (0.0,1.0)] # Sane as BK

for input_asym in input_asym:
    temp_control_acc = []
    temp_right_pert_acc = []
    temp_left_pert_acc = []
    for seed in range(10):
        path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/train_type_modular/n_neurons_256_random_seed_{}/n_epochs_10_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(seed, input_asym[0], input_asym[1])
        results_dict = np.load(os.path.join(path, 'all_val_results_dict.npy'), allow_pickle=True)
        temp_control_acc = np.mean([results_dict[-1]['control']['readout_accuracy_left'], results_dict[-1]['control']['readout_accuracy_right']])
        temp_right_pert_acc = np.mean([results_dict[-1]['right_alm_pert']['readout_accuracy_left'], results_dict[-1]['right_alm_pert']['readout_accuracy_right']])
        temp_left_pert_acc = np.mean([results_dict[-1]['left_alm_pert']['readout_accuracy_left'], results_dict[-1]['left_alm_pert']['readout_accuracy_right']])
    
    control_acc.append(np.mean(temp_control_acc))
    right_pert_acc.append(np.mean(temp_right_pert_acc))
    left_pert_acc.append(np.mean(temp_left_pert_acc))
    control_std.append(np.std(temp_control_acc))
    right_pert_std.append(np.std(temp_right_pert_acc))
    left_pert_std.append(np.std(temp_left_pert_acc))

xlabels=[-1, -0.8, -0.6, -0.2, 0, 0.2, 0.6, 0.8, 1]
plt.errorbar(xlabels, control_acc, yerr=control_std, label='Control', color='black', ls='--')
plt.errorbar(xlabels, right_pert_acc, yerr=right_pert_std, label='Right pert', color='darkgrey')
plt.errorbar(xlabels, left_pert_acc, yerr=left_pert_std, label='Left pert', color='lightgrey')
plt.ylabel('Readout accuracy')
plt.xlabel('Input asymmetry')
plt.ylim(0.5, 1)
plt.xticks([-1,0,1],[-1,0,1])
plt.legend()
plt.savefig('figs/frac_correct_diff_asymm.pdf')
plt.show()


