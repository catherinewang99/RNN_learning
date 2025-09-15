import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import json
import os

from dual_alm_rnn_models import *
from dual_alm_rnn_exp import DualALMRNNExp

plt.rcParams['pdf.fonttype'] = '42' 

def get_cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

recurrent_optimal = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)


# check how much the recurrent weights are aligned with the optimal solution
# Create a 2x5 subplot grid for 10 seeds
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.flatten()  # Flatten to 1D array for easier indexing

final_cosine_similarities_l, final_cosine_similarities_r = [], []
final_readout_weights_l, final_readout_weights_r = [], []

for seed in range(10):
    exp = DualALMRNNExp()
    exp.configs['random_seed'] = seed
    exp.configs['xs_left_alm_amp'] = 1.0
    exp.configs['xs_right_alm_amp'] = 1.0
    exp.init_sub_path(exp.configs['train_type'])
    logs_path = os.path.join(exp.configs['logs_dir'], exp.configs['model_type'], exp.sub_path)

    all_cosine_similarities_l, all_cosine_similarities_r = [], []
    for epoch in range(40):
        for bs in range(14):
            weights_dict = np.load(os.path.join(logs_path, 'weights_epoch_{}_within_hemi_{}.npz'.format(epoch, bs)))
            weights = weights_dict.get('rnn_cell.w_hh_linear_ll.weight').flatten()
            all_cosine_similarities_l.append(get_cosine_similarity(weights, recurrent_optimal))
            weights = weights_dict.get('rnn_cell.w_hh_linear_rr.weight').flatten()
            all_cosine_similarities_r.append(get_cosine_similarity(weights, recurrent_optimal))

    final_cosine_similarities_l.append(all_cosine_similarities_l[-1])
    final_cosine_similarities_r.append(all_cosine_similarities_r[-1])

    weights_dict = np.load(os.path.join(logs_path, 'weights_epoch_39_within_hemi_13.npz'))
    weights = weights_dict.get('readout_linear.weight').flatten()
    final_readout_weights_l.append(np.mean(weights[:2]))
    weights = weights_dict.get('readout_linear.weight').flatten()
    final_readout_weights_r.append(np.mean(weights[2:]))

    # Plot on the corresponding subplot
    axes[seed].plot(all_cosine_similarities_l, 'r', label='Left Hemisphere')
    axes[seed].plot(all_cosine_similarities_r, 'b', label='Right Hemisphere')
    axes[seed].set_title(f'Seed {seed}')
    axes[seed].set_xlabel('Epoch')
    axes[seed].set_ylabel('Cosine Similarity')
    axes[seed].grid(True, alpha=0.3)
    axes[seed].set_ylim(-1, 1)  # Cosine similarity ranges from -1 to 1

# Add a single legend for all subplots
fig.legend(['Left Hemisphere', 'Right Hemisphere'], loc='upper right', bbox_to_anchor=(0.98, 0.98))

# Add overall title
fig.suptitle('Alignment of recurrent weights to optimal solution over training (Left Hemisphere: {}, Right Hemisphere: {})'.format(exp.configs['xs_left_alm_amp'], exp.configs['xs_right_alm_amp']), fontsize=16, fontweight='bold')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined plot
plt.savefig('figs/cosine_similarity_across_learning_all_seeds_L{}_R{}.pdf'.format(exp.configs['xs_left_alm_amp'], exp.configs['xs_right_alm_amp']), bbox_inches='tight')
plt.show()


# plt.scatter(final_cosine_similarities_l, final_readout_weights_l, label='Left Hemisphere', color='r')
# plt.scatter(final_cosine_similarities_r, final_readout_weights_r, label='Right Hemisphere', color='b')
# plt.xlabel('Cosine Similarity')
# plt.ylabel('Readout Weight')
# plt.title('Final Cosine Similarity vs Readout Weight')
# plt.savefig('figs/final_cosine_similarity_vs_readout_weight_L{}_R{}.pdf'.format(exp.configs['xs_left_alm_amp'], exp.configs['xs_right_alm_amp']), bbox_inches='tight')
# plt.legend()
# plt.show()

plt.scatter(np.array(final_cosine_similarities_l)-np.array(final_cosine_similarities_r), np.array(final_readout_weights_l)-np.array(final_readout_weights_r))
plt.xlabel('Cosine Similarity Difference (Left - Right)')
plt.ylabel('Readout Weight Difference (Left - Right)')
plt.title('Final Cosine Similarity Difference vs Readout Weight Ratio')
plt.savefig('figs/final_cosine_similarity_difference_vs_readout_weight_difference_L{}_R{}.pdf'.format(exp.configs['xs_left_alm_amp'], exp.configs['xs_right_alm_amp']), bbox_inches='tight')
# plt.legend()
plt.show()


# Actually implement the optimal solution for the recurrent weights to check that it performs well and is stable


