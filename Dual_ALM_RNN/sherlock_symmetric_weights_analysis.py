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


for unfix_epoch in [1, 11, 40]:

    os.makedirs('figs/epoch_{}'.format(unfix_epoch), exist_ok=True)

    all_readout_bias = []
    all_readout_weights = []
    all_recurrent_weights = []
    all_recurrent_bias = []
    all_learning_epoch = []

    for seed in range(100):
        model_path = '/home/users/cwang314/dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/train_type_asymmetric_fixed_input_mult_seeds/n_neurons_4_random_seed_{}/unfix_epoch_{}/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_1.00/init_cross_hemi_rel_factor_0.20'
        checkpoint_file = 'model_epoch_0.pth'
        checkpoint_path = os.path.join(model_path.format(seed, unfix_epoch), checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        
        model = getattr(sys.modules[__name__], configs['model_type'])(configs, \
            exp.a, exp.pert_begin, exp.pert_end, noise=True)

        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)

        # get the relevant weights
        readout_bias = model.readout_linear.bias.data.cpu().numpy()[0]
        readout_weights = model.readout_linear.weight.data.cpu().numpy()
        recurrent_weights = model.rnn_cell.w_hh_linear_rr.weight.data.cpu().numpy()
        recurrent_bias = model.rnn_cell.w_hh_linear_rr.bias.data.cpu().numpy()
        print(recurrent_weights)

        # get learning epoch
        logs_path = '/home/users/cwang314/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_asymmetric_fixed_input_mult_seeds/n_neurons_4_random_seed_{}/unfix_epoch_{}/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_1.00/init_cross_hemi_rel_factor_0.20'
        results_path = os.path.join(logs_path.format(seed, unfix_epoch), 'all_val_results_dict.npy')
        if not os.path.exists(results_path):
            continue
        results_dict = np.load(results_path, allow_pickle=True)
        readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])
        if len(np.where(readout_acc_right >= 0.7)[0]) == 0:
            learning_epoch = 40
        else:
            learning_epoch = np.where(readout_acc_right >= 0.7)[0][0]

        all_readout_bias.append(readout_bias)
        all_readout_weights.append(readout_weights)
        all_recurrent_weights.append(recurrent_weights)
        all_recurrent_bias.append(recurrent_bias)
        all_learning_epoch.append(learning_epoch)

    np.save('figs/epoch_{}/all_readout_bias.npy'.format(unfix_epoch), all_readout_bias)
    np.save('figs/epoch_{}/all_readout_weights.npy'.format(unfix_epoch), all_readout_weights)
    np.save('figs/epoch_{}/all_recurrent_weights.npy'.format(unfix_epoch), all_recurrent_weights)
    np.save('figs/epoch_{}/all_recurrent_bias.npy'.format(unfix_epoch), all_recurrent_bias)
    np.save('figs/epoch_{}/all_learning_epoch.npy'.format(unfix_epoch), all_learning_epoch)