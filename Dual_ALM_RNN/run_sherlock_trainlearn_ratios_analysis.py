import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from numpy import linalg as LA


import argparse, os, math, pickle, json

# 1) Load your model
from dual_alm_rnn_models import *
from dual_alm_rnn_exp import DualALMRNNExp



exp = DualALMRNNExp()

# Load configs to get model parameters
with open('dual_alm_rnn_configs.json', 'r') as f:
    configs = json.load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # CW Mac update

switch_probs = np.array([[1,0,0,0],
               [0.91, 0.03, 0.03, 0.03],
               [0.7, 0.1, 0.1, 0.1,],
               [0.55, 0.15, 0.15, 0.15],
               [0.4, 0.2, 0.2, 0.2]])

# Create an outer dictionary to hold all results for each seed and cross_hemi
results = {}

for seed in range(50):
    results[seed] = {}
    for switch_prob in switch_probs:
        exp.configs['train_type'] = 'train_type_modular_fixed_input_cross_hemi_switch'
        exp.configs['switch_epoch_n'] = 0
        exp.configs['xs_left_alm_amp'] = 1.0
        exp.configs['xs_right_alm_amp'] = 1.0
        exp.configs['random_seed'] = seed
        exp.configs['unfix_epoch'] = 0
        exp.configs['init_cross_hemi_rel_factor'] = 0.5
        exp.configs['switch_ps'] = switch_prob
        exp.configs['bs'] = 256
        exp.configs['n_neurons'] = 256
        exp.configs['lr'] = 1e-4
        exp.configs['n_epochs'] = 45
        exp.configs['switch_epoch_n'] = 0

        model = getattr(sys.modules[__name__], configs['model_type'])(exp.configs, \
            exp.a, exp.pert_begin, exp.pert_end, noise=True)
        exp.init_sub_path(exp.configs['train_type'])


        checkpoint_file = 'all_val_results_dict.npy'
        checkpoint_path = os.path.join(exp.configs['logs_dir'], exp.configs['model_type'], exp.sub_path, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        results_dict = np.load(checkpoint_path, allow_pickle=True)

        epochs = np.arange(1, len(results_dict) + 1)
        readout_acc_left = np.array([results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))])
        readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])
        left_input_only_readout_acc_left = np.array([results_dict[i]['left_input_only']['readout_accuracy_left'] for i in range(len(results_dict))])
        left_input_only_readout_acc_right = np.array([results_dict[i]['left_input_only']['readout_accuracy_right'] for i in range(len(results_dict))])
        right_input_only_readout_acc_left = np.array([results_dict[i]['right_input_only']['readout_accuracy_left'] for i in range(len(results_dict))])
        right_input_only_readout_acc_right = np.array([results_dict[i]['right_input_only']['readout_accuracy_right'] for i in range(len(results_dict))])

        # Store results in outer dictionary
        results[seed][switch_prob[0]] = {
            "readout_acc_left": readout_acc_left,
            "readout_acc_right": readout_acc_right,
            "left_input_only_readout_acc_left": left_input_only_readout_acc_left,
            "left_input_only_readout_acc_right": left_input_only_readout_acc_right,
            "right_input_only_readout_acc_left": right_input_only_readout_acc_left,
            "right_input_only_readout_acc_right": right_input_only_readout_acc_right,

        }

np.save('results/sherlock_trainlearn_ratios_analysis.npy', results)






