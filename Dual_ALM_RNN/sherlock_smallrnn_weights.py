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

os.makedirs('figs/small_rnn_weights', exist_ok=True)

all_readout_bias_initial = []
all_readout_bias_final = []
all_readout_weights_initial = []
all_readout_weights_final = []
all_recurrent_weights_r_initial = []
all_recurrent_weights_r_final = []
all_recurrent_weights_l_initial = []
all_recurrent_weights_l_final = []
all_recurrent_bias_l_initial = []
all_recurrent_bias_l_final = []
all_recurrent_bias_r_initial = []
all_recurrent_bias_r_final = []

all_readout_acc_right = []
all_readout_acc_left = []

for seed in range(100):
    model_path = '/home/users/cwang314/dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/train_type_modular_fixed_input_cross_hemi_mult_seeds/n_neurons_4_random_seed_{}/unfix_epoch_40/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_1.00/init_cross_hemi_rel_factor_0.20'
    checkpoint_file_initial = 'model_epoch_0.pth'
    checkpoint_file_final = 'last_model.pth'
    checkpoint_path_initial = os.path.join(model_path.format(seed), checkpoint_file_initial)
    checkpoint_path_final = os.path.join(model_path.format(seed), checkpoint_file_final)
    if not os.path.exists(checkpoint_path_initial):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path_initial}")
    if not os.path.exists(checkpoint_path_final):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path_final}")

    configs['n_neurons'] = 4
    configs['random_seed'] = seed
    configs['unfix_epoch'] = 40
    configs['n_epochs'] = 40
    configs['lr'] = 3e-3
    configs['bs'] = 75

    
    model = getattr(sys.modules[__name__], configs['model_type'])(configs, \
        exp.a, exp.pert_begin, exp.pert_end, noise=True)

    state_dict = torch.load(checkpoint_path_initial, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    # get the relevant weights : want to save left and right weights and biases and also cross-hemispheric weights and biases
    readout_bias_initial = model.readout_linear.bias.data.cpu().numpy()[0]
    readout_weights_initial = model.readout_linear.weight.data.cpu().numpy()
    recurrent_weights_r_initial = model.rnn_cell.w_hh_linear_rr.weight.data.cpu().numpy()
    recurrent_bias_l_initial = model.rnn_cell.w_hh_linear_ll.bias.data.cpu().numpy()
    recurrent_weights_l_initial = model.rnn_cell.w_hh_linear_ll.weight.data.cpu().numpy()
    recurrent_bias_r_initial = model.rnn_cell.w_hh_linear_rr.bias.data.cpu().numpy()

    state_dict = torch.load(checkpoint_path_final, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    readout_bias_final = model.readout_linear.bias.data.cpu().numpy()[0]
    readout_weights_final = model.readout_linear.weight.data.cpu().numpy()
    recurrent_weights_r_final = model.rnn_cell.w_hh_linear_rr.weight.data.cpu().numpy()
    recurrent_bias_l_final = model.rnn_cell.w_hh_linear_ll.bias.data.cpu().numpy()
    recurrent_weights_l_final = model.rnn_cell.w_hh_linear_ll.weight.data.cpu().numpy()
    recurrent_bias_r_final = model.rnn_cell.w_hh_linear_rr.bias.data.cpu().numpy()

    # get learning epoch
    logs_path = '/home/users/cwang314/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_fixed_input_cross_hemi_mult_seeds/n_neurons_4_random_seed_{}/unfix_epoch_40/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_1.00/init_cross_hemi_rel_factor_0.20'
    results_path = os.path.join(logs_path.format(seed), 'all_val_results_dict.npy')
    if not os.path.exists(results_path):
        continue
    results_dict = np.load(results_path, allow_pickle=True)
    readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])
    readout_acc_left = np.array([results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))])

    all_readout_acc_right.append(readout_acc_right)
    all_readout_acc_left.append(readout_acc_left)

    all_readout_bias_initial.append(readout_bias_initial)
    all_readout_bias_final.append(readout_bias_final)
    all_readout_weights_initial.append(readout_weights_initial)
    all_readout_weights_final.append(readout_weights_final)
    all_recurrent_weights_r_initial.append(recurrent_weights_r_initial)
    all_recurrent_weights_r_final.append(recurrent_weights_r_final)
    all_recurrent_weights_l_initial.append(recurrent_weights_l_initial)
    all_recurrent_weights_l_final.append(recurrent_weights_l_final)
    all_recurrent_bias_l_initial.append(recurrent_bias_l_initial)
    all_recurrent_bias_l_final.append(recurrent_bias_l_final)
    all_recurrent_bias_r_initial.append(recurrent_bias_r_initial)
    all_recurrent_bias_r_final.append(recurrent_bias_r_final)

np.save('figs/small_rnn_weights/all_readout_bias_initial.npy', all_readout_bias_initial)
np.save('figs/small_rnn_weights/all_readout_bias_final.npy', all_readout_bias_final)
np.save('figs/small_rnn_weights/all_readout_weights_initial.npy', all_readout_weights_initial)
np.save('figs/small_rnn_weights/all_readout_weights_final.npy', all_readout_weights_final)
np.save('figs/small_rnn_weights/all_recurrent_weights_r_initial.npy', all_recurrent_weights_r_initial)
np.save('figs/small_rnn_weights/all_recurrent_weights_r_final.npy', all_recurrent_weights_r_final)
np.save('figs/small_rnn_weights/all_recurrent_weights_l_initial.npy', all_recurrent_weights_l_initial)
np.save('figs/small_rnn_weights/all_recurrent_weights_l_final.npy', all_recurrent_weights_l_final)
np.save('figs/small_rnn_weights/all_recurrent_bias_l_initial.npy', all_recurrent_bias_l_initial)
np.save('figs/small_rnn_weights/all_recurrent_bias_l_final.npy', all_recurrent_bias_l_final)
np.save('figs/small_rnn_weights/all_recurrent_bias_r_initial.npy', all_recurrent_bias_r_initial)
np.save('figs/small_rnn_weights/all_recurrent_bias_r_final.npy', all_recurrent_bias_r_final)

np.save('figs/small_rnn_weights/all_readout_acc_right.npy', all_readout_acc_right)
np.save('figs/small_rnn_weights/all_readout_acc_left.npy', all_readout_acc_left)