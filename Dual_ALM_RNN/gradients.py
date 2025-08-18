import numpy as np
import matplotlib.pyplot as plt 
from dual_alm_rnn_exp import DualALMRNNExp
import json
import os




plt.rcParams['pdf.fonttype'] = '42' 
exp = DualALMRNNExp()


with open('dual_alm_rnn_configs.json','r') as read_file:
    configs = json.load(read_file)

if configs['one_hot'] and configs['train_type'] == "train_type_modular_corruption":
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/{}/onehot_cor_type_{}_epoch_{}_noise_{:.2f}/n_neurons_256_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['corruption_type'], configs['corruption_start_epoch'], configs['corruption_noise'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])
elif configs['one_hot'] and configs['train_type'] == "train_type_modular":
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/{}/onehot/n_neurons_256_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])
elif configs['train_type'] == "train_type_modular_corruption":
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/{}/cor_type_{}_epoch_{}_noise_{:.2f}/n_neurons_256_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['corruption_type'], configs['corruption_start_epoch'], configs['corruption_noise'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])
else:
    path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/{}/n_neurons_256_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_{}/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(configs['train_type'], configs['random_seed'], configs['n_epochs'], configs['across_hemi_n_epochs'], configs['xs_left_alm_amp'], configs['xs_right_alm_amp'])


# Get all weight files
weight_files = [f for f in os.listdir(path) if f.startswith('weights_epoch_')]
weight_files.sort()

# Plot each of the r to r weight matrices (w_hh_linear_rr) as 10 subplots, each as a heatmap sharing the same colormap


# If the weights are stored in 'weights_epoch_{}.npy' files, load w_hh_linear_rr from each
rr_weights = []
for f in weight_files:  # only first 10 epochs
    data = np.load(os.path.join(path, f))
    # If data is a dict with keys, extract 'w_hh_linear_rr'
    rr_weights.append(data['w_hh_linear_rr'])

# Determine global vmin/vmax for colormap
all_min = min([w.min() for w in rr_weights])
all_max = max([w.max() for w in rr_weights])

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i in range(10):
    ax = axes[i]
    if i < len(rr_weights):
        im = ax.imshow(rr_weights[i], vmin=all_min, vmax=all_max, cmap='bwr', aspect='auto')
        ax.set_title(f'Epoch {i+1}')
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Neuron')
    else:
        ax.axis('off')

# Add a single colorbar for all subplots
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Weight Value')

fig.suptitle('figs/w_hh_linear_rr Weight Matrices Across Epochs', fontsize=18)
plt.tight_layout(rect=[0, 0, 0.88, 1])
fig.savefig('figs/w_hh_linear_rr_heatmaps.pdf', dpi=300, bbox_inches='tight')
plt.show()


# Plot heatmaps of the difference (delta) between w_hh_linear_rr weights at consecutive epochs

# Compute deltas: rr_weights[n+1] - rr_weights[n] for n in 0..len(rr_weights)-2
rr_deltas = [rr_weights[i+1] - rr_weights[i] for i in range(len(rr_weights)-1)]

# Determine global vmin/vmax for colormap across all deltas
delta_min = min([d.min() for d in rr_deltas])
delta_max = max([d.max() for d in rr_deltas])
abs_max = max(abs(delta_min), abs(delta_max))  # for symmetric colormap

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i in range(10):
    ax = axes[i]
    if i < len(rr_deltas):
        im = ax.imshow(rr_deltas[i], vmin=-abs_max, vmax=abs_max, cmap='bwr', aspect='auto')
        ax.set_title(f'Epoch {i+1} → {i+2}')
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Neuron')
    else:
        ax.axis('off')

# Add a single colorbar for all subplots
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Δ Weight Value')

fig.suptitle('Δ w_hh_linear_rr (Epoch n+1 - Epoch n)', fontsize=18)
plt.tight_layout(rect=[0, 0, 0.88, 1])
# Save the figure with one_hot and train_type info in the filename
fig.savefig(f"figs/w_hh_linear_rr_delta_heatmaps_onehot_{configs['one_hot']}_train_{configs['train_type']}.pdf", dpi=300, bbox_inches='tight')

plt.show()



import numpy.linalg as la

# Perform SVD on each delta matrix and store singular values and first singular vectors
svd_results = []
for i, delta in enumerate(rr_deltas):
    # SVD: delta = U S V^T
    U, S, Vh = la.svd(delta, full_matrices=False)
    svd_results.append({
        'U': U,
        'S': S,
        'Vh': Vh,
        'epoch_pair': (i+1, i+2)
    })

# Plot singular values across epochs
fig_sv, ax_sv = plt.subplots(figsize=(8, 5))
for i, res in enumerate(svd_results):
    ax_sv.plot(res['S'], 'o-', label=f'Epoch {res["epoch_pair"][0]}→{res["epoch_pair"][1]}')
ax_sv.set_xlabel('Singular Value Index')
ax_sv.set_ylabel('Singular Value')
ax_sv.set_title('Singular Values of Δ w_hh_linear_rr Across Epochs')
ax_sv.legend(fontsize=8, ncol=2)
ax_sv.grid(True, alpha=0.3)
plt.tight_layout()
fig_sv.savefig(f"figs/w_hh_linear_rr_delta_singular_values_onehot_{configs['one_hot']}_train_{configs['train_type']}.pdf", dpi=300, bbox_inches='tight')

# Compare first left singular vector (U[:,0]) across epochs
fig_u, ax_u = plt.subplots(figsize=(10, 6))
for i, res in enumerate(svd_results):
    ax_u.plot(res['U'][:, 0], label=f'Epoch {res["epoch_pair"][0]}→{res["epoch_pair"][1]}')
ax_u.set_xlabel('Neuron Index')
ax_u.set_ylabel('First Left Singular Vector (U[:,0])')
ax_u.set_title('First Left Singular Vector of Δ w_hh_linear_rr Across Epochs')
ax_u.legend(fontsize=8, ncol=2)
ax_u.grid(True, alpha=0.3)
plt.tight_layout()
fig_u.savefig(f"figs/w_hh_linear_rr_delta_first_left_singular_vector_onehot_{configs['one_hot']}_train_{configs['train_type']}.pdf", dpi=300, bbox_inches='tight')

# Compare first right singular vector (Vh[0,:]) across epochs
fig_v, ax_v = plt.subplots(figsize=(10, 6))
for i, res in enumerate(svd_results):
    ax_v.plot(res['Vh'][0, :], label=f'Epoch {res["epoch_pair"][0]}→{res["epoch_pair"][1]}')
ax_v.set_xlabel('Neuron Index')
ax_v.set_ylabel('First Right Singular Vector (Vh[0,:])')
ax_v.set_title('First Right Singular Vector of Δ w_hh_linear_rr Across Epochs')
ax_v.legend(fontsize=8, ncol=2)
ax_v.grid(True, alpha=0.3)
plt.tight_layout()
fig_v.savefig(f"figs/w_hh_linear_rr_delta_first_right_singular_vector_onehot_{configs['one_hot']}_train_{configs['train_type']}.pdf", dpi=300, bbox_inches='tight')

plt.show()


