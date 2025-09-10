"""
Plot the gradients of the small RNN along the readout  for different trial types / corruption conditions
"""

import numpy as np
import matplotlib.pyplot as plt 
from dual_alm_rnn_exp import DualALMRNNExp
import json
import os
from dual_alm_rnn_models import *
from dual_alm_rnn_exp import DualALMRNNExp

plt.rcParams['pdf.fonttype'] = '42' 
sensory_inputs = np.load('dual_alm_rnn_data/test/onehot_sensory_inputs_simple.npy')
trial_type_labels = np.load('dual_alm_rnn_data/test/onehot_trial_type_labels_simple.npy')

# visualize the projected gradients on a version of the model with artificial inputs








# Visualize the gradients derived from training

exp = DualALMRNNExp()

if exp.configs['train_type'] == 'train_type_modular_corruption':
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_corruption/onehot_cor_type_{}_epoch_{}_noise_{:.2f}/n_neurons_4_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/'.format(exp.configs['corruption_type'], exp.configs['corruption_start_epoch'], exp.configs['corruption_noise'], exp.configs['random_seed'], exp.configs['n_epochs'], exp.configs['xs_left_alm_amp'], exp.configs['xs_right_alm_amp'])
elif exp.configs['train_type'] == 'train_type_modular_single_readout':
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_single_readout/n_neurons_4_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/'.format(exp.configs['random_seed'], exp.configs['n_epochs'], exp.configs['xs_left_alm_amp'], exp.configs['xs_right_alm_amp'])

all_readout_weights, all_readout_bias = [], []
all_readout_gradients_l, all_readout_gradients_r = [], []
all_recurrent_gradients_l, all_recurrent_gradients_r = [], []
all_input_gradients_l, all_input_gradients_r = [], []
for epoch in range(40):
    for batch_idx in range(13):

        # weights= np.load(os.path.join(path, 'weights_epoch_{}_within_hemi_{}.npz'.format(epoch, batch_idx)))
        # readout_weights = weights.get('readout_linear.weight')[0]
        # readout_bias = weights.get('readout_linear.bias')[0]


        gradients = np.load(os.path.join(path, 'gradients_epoch_{}_within_hemi_{}.npz'.format(epoch, batch_idx)))
        readout_gradients = gradients.get('readout_linear.weight')[0]

        # all_readout_weights.append(readout_weights)
        # all_readout_bias.append(readout_bias)
        all_readout_gradients_l.append(readout_gradients[:2])
        all_readout_gradients_r.append(readout_gradients[2:])
        all_recurrent_gradients_l.append(gradients.get('rnn_cell.w_hh_linear_ll.weight').flatten())
        all_recurrent_gradients_r.append(gradients.get('rnn_cell.w_hh_linear_rr.weight').flatten())
        all_input_gradients_l.append(gradients.get('w_xh_linear_left_alm.weight').flatten())
        all_input_gradients_r.append(gradients.get('w_xh_linear_right_alm.weight').flatten())





f=plt.figure(figsize=(10,7))
# plt.plot(all_readout_weights, 'b')
# plt.plot(all_readout_bias, 'r')
plt.plot(all_readout_gradients_l, 'r', label='Left Hemisphere')
plt.plot(all_readout_gradients_r, 'b', label='Right Hemisphere')
for i in range(40):
    plt.axvline(i*13, color='k', linestyle='--', alpha=0.5)
if exp.configs['train_type'] == 'train_type_modular_corruption':
    # Add faded red background for epochs after corruption starts
    corruption_start_epoch = exp.configs['corruption_start_epoch']
    n_batches_per_epoch = 13
    plt.axvspan(corruption_start_epoch * n_batches_per_epoch, 40 * n_batches_per_epoch, color='red', alpha=0.15, zorder=-1)
plt.title('Gradients of Readout Weights Over Training')
plt.xlabel('Batch Index')
plt.ylabel('Gradient')
plt.legend()
plt.savefig('figs/readout_gradients_small_rnn_{}.pdf'.format(exp.configs['train_type']), dpi=300, bbox_inches='tight')
plt.show()

f=plt.figure(figsize=(10,7))
# plt.plot(all_readout_weights, 'b')
# plt.plot(all_readout_bias, 'r')
plt.plot(np.log(np.abs(all_recurrent_gradients_l)), 'r', label='Left Hemisphere')
plt.plot(np.log(np.abs(all_recurrent_gradients_r)), 'b', label='Right Hemisphere')
for i in range(40):
    plt.axvline(i*13, color='k', linestyle='--', alpha=0.5)
if exp.configs['train_type'] == 'train_type_modular_corruption':
    # Add faded red background for epochs after corruption starts
    corruption_start_epoch = exp.configs['corruption_start_epoch']
    n_batches_per_epoch = 13
    plt.axvspan(corruption_start_epoch * n_batches_per_epoch, 40 * n_batches_per_epoch, color='red', alpha=0.15, zorder=-1)
plt.title('Gradients of Recurrent Weights Over Training')
plt.xlabel('Batch Index')
plt.ylabel('Log abs Gradient')
plt.legend()
plt.savefig('figs/recurrent_gradients_small_rnn_{}.pdf'.format(exp.configs['train_type']), dpi=300, bbox_inches='tight')
plt.show()

f=plt.figure(figsize=(10,7))
# plt.plot(all_readout_weights, 'b')
# plt.plot(all_readout_bias, 'r')
plt.plot(np.log(np.abs(all_input_gradients_l)), 'r', label='Left Hemisphere')
plt.plot(np.log(np.abs(all_input_gradients_r)), 'b', label='Right Hemisphere')
for i in range(40):
    plt.axvline(i*13, color='k', linestyle='--', alpha=0.5)
if exp.configs['train_type'] == 'train_type_modular_corruption':
    # Add faded red background for epochs after corruption starts
    corruption_start_epoch = exp.configs['corruption_start_epoch']
    n_batches_per_epoch = 13
    plt.axvspan(corruption_start_epoch * n_batches_per_epoch, 40 * n_batches_per_epoch, color='red', alpha=0.15, zorder=-1)
plt.title('Gradients of Input Weights Over Training')
plt.xlabel('Batch Index')
plt.ylabel('Log abs Gradient')
plt.legend()
plt.savefig('figs/input_gradients_small_rnn_{}.pdf'.format(exp.configs['train_type']), dpi=300, bbox_inches='tight')
plt.show()

plt.bar([0,1,2,3,4,5], [np.mean(np.abs(all_readout_gradients_l)), np.mean(np.abs(all_readout_gradients_r)), np.mean(np.abs(all_recurrent_gradients_l)), np.mean(np.abs(all_recurrent_gradients_r)), np.mean(np.abs(all_input_gradients_l)), np.mean(np.abs(all_input_gradients_r))])
plt.xticks([0,1,2,3,4,5], ['Readout Left', 'Readout Right', 'Recurrent Left', 'Recurrent Right', 'Input Left', 'Input Right'])
plt.ylabel('Average absGradient')
plt.title('Average Gradients of Weights Over Training')
plt.savefig('figs/gradients_small_rnn_{}.pdf'.format(exp.configs['train_type']), dpi=300, bbox_inches='tight')
plt.show()