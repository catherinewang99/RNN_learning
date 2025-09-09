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

all_readout_weights, all_readout_bias = [], []
all_readout_gradients_l, all_readout_gradients_r = [], []
for epoch in range(40):
    for batch_idx in range(13):

        weights= np.load('/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_single_readout/n_neurons_4_random_seed_0/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.40/init_cross_hemi_rel_factor_0.20/weights_epoch_{}_within_hemi_{}.npz'.format(epoch, batch_idx))
        readout_weights = weights.get('readout_linear.weight')[0]
        readout_bias = weights.get('readout_linear.bias')[0]


        gradients = np.load('/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_single_readout/n_neurons_4_random_seed_0/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.40/init_cross_hemi_rel_factor_0.20/gradients_epoch_{}_within_hemi_{}.npz'.format(epoch, batch_idx))
        readout_gradients = gradients.get('readout_linear.weight')[0]

        all_readout_weights.append(readout_weights)
        all_readout_bias.append(readout_bias)
        all_readout_gradients_l.append(readout_gradients[:2])
        all_readout_gradients_r.append(readout_gradients[2:])





# plt.plot(all_readout_weights, 'b')
# plt.plot(all_readout_bias, 'r')
plt.plot(all_readout_gradients_l, 'r')
plt.plot(all_readout_gradients_r, 'b')
for i in range(40):
    plt.axvline(i*13, color='k', linestyle='--', alpha=0.5)
plt.title('Gradients of Readout Weights Over Training')
plt.xlabel('Batch Index')
plt.ylabel('Gradient')
plt.legend(['Left Hemisphere', 'Right Hemisphere'])
plt.savefig('figs/gradients_small_rnn.pdf', dpi=300, bbox_inches='tight')
plt.show()
