import numpy as np
import matplotlib.pyplot as plt 
import os
plt.rcParams['pdf.fonttype'] = '42' 

# Load all_val_results_dict.npy
results_dict = np.load('all_val_results_dict.npy', allow_pickle=True)

# Single seed

path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/train_type_modular/n_neurons_256_random_seed_1/n_epochs_10_n_epochs_across_hemi_10/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.20/init_cross_hemi_rel_factor_0.20'
# Load input_weights_left_epoch_0.npy and input_weights_right_epoch_0.npy

input_weights_left_epoch = np.load(os.path.join(path, 'input_weights_left_epoch_final.npy'))
input_weights_right_epoch = np.load(os.path.join(path, 'input_weights_right_epoch_final.npy'))
readout_weights_left_epoch = np.load(os.path.join(path, 'readout_weights_left_epoch_final.npy'))
readout_weights_right_epoch = np.load(os.path.join(path, 'readout_weights_right_epoch_final.npy'))


plt.figure()

# Plot input weights for left and right ALM


plt.scatter(input_weights_left_epoch, readout_weights_left_epoch, color='r', alpha=0.5)
plt.scatter(input_weights_right_epoch, readout_weights_right_epoch, color='b', alpha=0.5)
plt.xlabel('Input weights')
plt.ylabel('Readout weights')
plt.title('Sensory input vs readout random init {}'.format(1))
plt.savefig('figs/sensory_input_vs_readout_random_init_{}.pdf'.format(1))
plt.show()


# Compute correlation between input and readout weights for left ALM
corr_left = np.corrcoef(input_weights_left_epoch, readout_weights_left_epoch)[0, 1]
print(f"Correlation between input and readout weights (Left ALM): {corr_left:.3f}")

# Compute correlation between input and readout weights for right ALM
corr_right = np.corrcoef(input_weights_right_epoch, readout_weights_right_epoch)[0, 1]
print(f"Correlation between input and readout weights (Right ALM): {corr_right:.3f}")


#look at multiple seeds
# Plotting input vs readout weights across random seeds 0 to 24 (5x5 grid)
fig, axes = plt.subplots(5, 5, figsize=(15, 15), sharex=True, sharey=True)

for seed in range(25):
    row, col = divmod(seed, 5)
    path_seed = f'/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/train_type_modular/n_neurons_256_random_seed_{seed}/n_epochs_10_n_epochs_across_hemi_10/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.20/init_cross_hemi_rel_factor_0.20'
    try:
        input_weights_left = np.load(os.path.join(path_seed, 'input_weights_left_epoch_1.npy'))
        input_weights_right = np.load(os.path.join(path_seed, 'input_weights_right_epoch_1.npy'))
        readout_weights_left = np.load(os.path.join(path_seed, 'readout_weights_left_epoch_1.npy'))
        readout_weights_right = np.load(os.path.join(path_seed, 'readout_weights_right_epoch_1.npy'))
    except Exception as e:
        print(f"Seed {seed}: Could not load weights. Skipping. Error: {e}")
        continue

    ax = axes[row, col]
    # Overlay left (red) and right (blue) on the same subplot
    ax.scatter(input_weights_left, readout_weights_left, color='r', alpha=0.5, label='Left ALM' if seed == 0 else "")
    ax.scatter(input_weights_right, readout_weights_right, color='b', alpha=0.5, label='Right ALM' if seed == 0 else "")
    corr_left = np.corrcoef(input_weights_left, readout_weights_left)[0, 1]
    corr_right = np.corrcoef(input_weights_right, readout_weights_right)[0, 1]
    ax.set_title(f"Seed {seed}\nL: {corr_left:.2f}  R: {corr_right:.2f}", fontsize=8)
    if row == 4:
        ax.set_xlabel('Input weights', fontsize=8)
    if col == 0:
        ax.set_ylabel('Readout weights', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)
    if seed == 0:
        ax.legend(fontsize=7, loc='best')

fig.suptitle('Sensory input vs readout weights across seeds 0-24\nRed: Left ALM, Blue: Right ALM', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('figs/sensory_input_vs_readout_overlay_0_24.pdf')
plt.show()


