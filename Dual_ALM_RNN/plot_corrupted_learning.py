import numpy as np
import matplotlib.pyplot as plt 
from dual_alm_rnn_exp import DualALMRNNExp
import seaborn as sns
import glob
import os


plt.rcParams['pdf.fonttype'] = '42' 

exp = DualALMRNNExp()
if exp.configs['one_hot']:
    results_dict = np.load('/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/train_type_modular_corruption/onehot_cor_type_{}_epoch_{}_noise_{}0/n_neurons_256_random_seed_{}/n_epochs_10_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.20/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(exp.configs['corruption_type'], exp.configs['corruption_start_epoch'], exp.configs['corruption_noise'], exp.configs['random_seed']), allow_pickle=True)
else:
    results_dict = np.load('/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/train_type_modular_corruption/cor_type_{}_epoch_{}_noise_{}0/n_neurons_256_random_seed_{}/n_epochs_10_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.20/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(exp.configs['corruption_type'], exp.configs['corruption_start_epoch'], exp.configs['corruption_noise'], exp.configs['random_seed']), allow_pickle=True)

# ONE SEED

# CD vs readout accuracy
f,ax=plt.subplots(1,2,figsize=(10,5), sharey='row')
for i in range(len(results_dict)):
    ax[0].scatter(results_dict[i]['control']['cd_accuracy_left'], results_dict[i]['control']['readout_accuracy_left'], color='r', alpha=0.5)
    ax[1].scatter(results_dict[i]['control']['cd_accuracy_right'], results_dict[i]['control']['readout_accuracy_right'], color='b', alpha=0.5)

ax[0].set_title('L hemi')
ax[1].set_title('R hemi')
ax[0].set_xlabel('CD Accuracy')
ax[0].set_ylabel('Readout Accuracy')
plt.suptitle('Control Accuracy')
plt.show()


# L vs R readout accuracy
f=plt.figure()
for i in range(len(results_dict)):
    plt.scatter(results_dict[i]['control']['readout_accuracy_left'], results_dict[i]['control']['readout_accuracy_right'], color='black', alpha=0.5)

# Compute min and max for the scatter plot axes
all_left = [results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))]
all_right = [results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))]
min_val = min(np.min(all_left), np.min(all_right))
max_val = max(np.max(all_left), np.max(all_right))
plt.plot([min_val, max_val], [min_val, max_val], 'k:', lw=2)  # add dotted diagonal line
plt.xlabel('Readout Accuracy Left')
plt.ylabel('Readout Accuracy Right')
plt.title('Control Accuracy')
plt.show()



# Plot readout accuracies and L/R agreement over learning in two vertically stacked subplots sharing x-axis

epochs = np.arange(1, len(results_dict) + 1)
readout_acc_left = np.array([results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))])
readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])

n_trials_agreed = np.array([results_dict[i]['control']['n_trials_agreed'] for i in range(len(results_dict))])
n_trials = np.array([results_dict[i]['control']['n_trials'] for i in range(len(results_dict))])
agreement_frac = n_trials_agreed / n_trials

# Compute chance level: p(agree) = p_L^2 + (1-p_L)^2 if p_L = p_R, but here use both
chance_agree = readout_acc_left * readout_acc_right + (1 - readout_acc_left) * (1 - readout_acc_right)

corruption_start_epoch = exp.configs['corruption_start_epoch']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Top subplot: Readout accuracies
ax1.plot(epochs, readout_acc_left, color='r', label='Left Hemi Readout Accuracy')
ax1.plot(epochs, readout_acc_right, color='b', label='Right Hemi Readout Accuracy')
ax1.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2, label='Corruption Start (Epoch {})'.format(corruption_start_epoch))
ax1.set_ylabel('Readout Accuracy (Control)')
ax1.set_title('Readout Accuracy Over Training (Control Trials)')
ax1.set_xticks(epochs)
ax1.set_ylim(0.4, 1.05)
ax1.legend()

# Bottom subplot: Agreement
ax2.plot(epochs, agreement_frac, color='k', label='Empirical Agreement (L=R)')
ax2.plot(epochs, chance_agree, color='gray', linestyle='--', label='Chance Level (Binomial)')
ax2.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2, label='Corruption Start (Epoch {})'.format(corruption_start_epoch))
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Fraction of Trials with L=R')
ax2.set_title('Left/Right ALM Agreement Over Training (Control Trials)')
ax2.set_xticks(epochs)
ax2.set_ylim(0.4, 1.05)
ax2.legend()

plt.tight_layout()
plt.savefig('figs/LR_readoutacc_and_agreement_corrupted_learning_epoch_{}_noise_{}0_type_{}.pdf'.format(exp.configs['corruption_start_epoch'], exp.configs['corruption_noise'], exp.configs['corruption_type']))
plt.show()




## Look at perturbation conditions


# Plot the performance at the end of training (last epoch) for all models where corruption started at epoch 7


# Preallocate arrays for each condition (one value per model/seed)
control_left = []
leftpert_left = []
rightpert_left = []

control_right = []
leftpert_right = []
rightpert_right = []


# Get the last epoch (post-training) for each condition
control_left.append(results_dict[-1]['control']['readout_accuracy_left'])
leftpert_left.append(results_dict[-1]['left_alm_pert']['readout_accuracy_left'])
rightpert_left.append(results_dict[-1]['right_alm_pert']['readout_accuracy_left'])
control_right.append(results_dict[-1]['control']['readout_accuracy_right'])
leftpert_right.append(results_dict[-1]['left_alm_pert']['readout_accuracy_right'])
rightpert_right.append(results_dict[-1]['right_alm_pert']['readout_accuracy_right'])

# Convert to numpy arrays
control_left = np.array(control_left)
leftpert_left = np.array(leftpert_left)
rightpert_left = np.array(rightpert_left)

control_right = np.array(control_right)
leftpert_right = np.array(leftpert_right)
rightpert_right = np.array(rightpert_right)

n_models = len(control_left)

# Prepare data for plotting
bar_means_left = [np.mean(control_left), np.mean(leftpert_left), np.mean(rightpert_left)]
bar_sems_left = [np.std(control_left, ddof=1)/np.sqrt(n_models),
                np.std(leftpert_left, ddof=1)/np.sqrt(n_models),
                np.std(rightpert_left, ddof=1)/np.sqrt(n_models)]

bar_means_right = [np.mean(control_right), np.mean(leftpert_right), np.mean(rightpert_right)]
bar_sems_right = [np.std(control_right, ddof=1)/np.sqrt(n_models),
                np.std(leftpert_right, ddof=1)/np.sqrt(n_models),
                np.std(rightpert_right, ddof=1)/np.sqrt(n_models)]

labels = ['Control', 'Left Perturb', 'Right Perturb']
bar_colors = ['#888888', '#e41a1c', '#377eb8']  # grey, red, blue

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# --- Left Hemi subplot ---
ax = axes[0]
bars = ax.bar(range(3), bar_means_left, yerr=bar_sems_left, capsize=6, color=bar_colors, alpha=0.7)

# Scatter the individual model points and connect them
for i, (c, l, r) in enumerate(zip(control_left, leftpert_left, rightpert_left)):
    ax.plot([0, 1, 2], [c, l, r], color='gray', alpha=0.5, marker='o', markersize=7, linewidth=1.5, zorder=10)

# Overlay the individual points for each bar
ax.scatter(np.full(n_models, 0), control_left, color='k', s=40, zorder=20, label='Models')
ax.scatter(np.full(n_models, 1), leftpert_left, color='k', s=40, zorder=20)
ax.scatter(np.full(n_models, 2), rightpert_left, color='k', s=40, zorder=20)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels)
ax.set_ylabel('Readout Accuracy')
ax.set_title('Left Hemi Readout Accuracy\n(Corruption@Epoch {})'.format(corruption_start_epoch))
ax.set_ylim(0.0, 1.05)
ax.grid(axis='y', linestyle=':', alpha=0.5)

# --- Right Hemi subplot ---
ax = axes[1]
bars = ax.bar(range(3), bar_means_right, yerr=bar_sems_right, capsize=6, color=bar_colors, alpha=0.7)

for i, (c, l, r) in enumerate(zip(control_right, leftpert_right, rightpert_right)):
    ax.plot([0, 1, 2], [c, l, r], color='gray', alpha=0.5, marker='o', markersize=7, linewidth=1.5, zorder=10)

ax.scatter(np.full(len(control_right), 0), control_right, color='k', s=40, zorder=20, label='Models')
ax.scatter(np.full(len(leftpert_right), 1), leftpert_right, color='k', s=40, zorder=20)
ax.scatter(np.full(len(rightpert_right), 2), rightpert_right, color='k', s=40, zorder=20)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels)
ax.set_title('Right Hemi Readout Accuracy\n(Corruption@Epoch {})'.format(corruption_start_epoch))
ax.set_ylim(0.0, 1.05)
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()



if False:

    ## Aggregate results over random seeds

    results_dict = np.load('/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/all_val_results_dict_corrupted.npy', allow_pickle=True)

    # Plot all corruption start epochs (1-10) in a large grid of subplots (2 per corruption epoch: accuracy, agreement)
    import scipy.stats as stats

    n_seeds = 10
    corruption_start_epochs = list(range(1, 10))  # 1 to 10
    n_epochs = 10  # assumed from training

    # Preallocate: for each corruption_start_epoch, store mean/sem arrays
    all_means = []
    all_sems = []
    all_epochs = []

    for corruption_start_epoch in corruption_start_epochs:
        all_readout_acc_left = []
        all_readout_acc_right = []
        all_agreement_frac = []
        all_chance_agree = []

        for seed in range(n_seeds):
            results_dict = np.load(
                '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/train_type_modular_corruption/corr_epoch_{}_noise_0.70/n_neurons_256_random_seed_{}/n_epochs_10_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.20/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'
                .format(corruption_start_epoch, seed), allow_pickle=True)

            epochs = np.arange(1, len(results_dict) + 1)
            readout_acc_left = np.array([results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))])
            readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])

            n_trials_agreed = np.array([results_dict[i]['control']['n_trials_agreed'] for i in range(len(results_dict))])
            n_trials = np.array([results_dict[i]['control']['n_trials'] for i in range(len(results_dict))])
            agreement_frac = n_trials_agreed / n_trials

            # Compute chance level: p(agree) = p_L^2 + (1-p_L)^2 if p_L = p_R, but here use both
            chance_agree = readout_acc_left * readout_acc_right + (1 - readout_acc_left) * (1 - readout_acc_right)

            all_readout_acc_left.append(readout_acc_left)
            all_readout_acc_right.append(readout_acc_right)
            all_agreement_frac.append(agreement_frac)
            all_chance_agree.append(chance_agree)

        # Convert to numpy arrays: shape (n_seeds, n_epochs)
        all_readout_acc_left = np.array(all_readout_acc_left)
        all_readout_acc_right = np.array(all_readout_acc_right)
        all_agreement_frac = np.array(all_agreement_frac)
        all_chance_agree = np.array(all_chance_agree)

        # Compute mean and SEM across seeds
        mean_readout_acc_left = np.mean(all_readout_acc_left, axis=0)
        sem_readout_acc_left = stats.sem(all_readout_acc_left, axis=0)
        mean_readout_acc_right = np.mean(all_readout_acc_right, axis=0)
        sem_readout_acc_right = stats.sem(all_readout_acc_right, axis=0)

        mean_agreement_frac = np.mean(all_agreement_frac, axis=0)
        sem_agreement_frac = stats.sem(all_agreement_frac, axis=0)
        mean_chance_agree = np.mean(all_chance_agree, axis=0)
        sem_chance_agree = stats.sem(all_chance_agree, axis=0)

        all_means.append({
            'readout_acc_left': mean_readout_acc_left,
            'readout_acc_right': mean_readout_acc_right,
            'agreement_frac': mean_agreement_frac,
            'chance_agree': mean_chance_agree
        })
        all_sems.append({
            'readout_acc_left': sem_readout_acc_left,
            'readout_acc_right': sem_readout_acc_right,
            'agreement_frac': sem_agreement_frac,
            'chance_agree': sem_chance_agree
        })
        all_epochs.append(np.arange(1, all_readout_acc_left.shape[1] + 1))

    # Plot: 9 corruption start epochs × 2 subplots (accuracy, agreement) = 18 subplots,
    # arranged as 4 rows: 1st and 3rd are accuracy, 2nd and 4th are agreement.
    # 5 columns on top (rows 0,1), 4 columns on bottom (rows 2,3).

    n_cols_top = 5
    n_cols_bottom = 4
    n_rows = 4
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols_top, figsize=(26, 14), sharex=True, sharey='row'
    )
    # axes shape: (4,5), but bottom row's last axis is unused

    for idx, corruption_start_epoch in enumerate(corruption_start_epochs):
        # Determine subplot position
        if idx < n_cols_top:
            # Top block: columns 0-4, rows 0 (acc), 1 (agree)
            col = idx
            row_acc = 0
            row_agree = 1
        else:
            # Bottom block: columns 0-3, rows 2 (acc), 3 (agree)
            col = idx - n_cols_top
            row_acc = 2
            row_agree = 3
            if col >= n_cols_bottom:
                continue  # skip if more than 9 epochs

        means = all_means[idx]
        sems = all_sems[idx]
        epochs = all_epochs[idx]

        # Accuracy subplot
        ax_acc = axes[row_acc, col]
        ax_acc.plot(epochs, means['readout_acc_left'], color='r', label='Left Hemi')
        ax_acc.fill_between(epochs, means['readout_acc_left'] - sems['readout_acc_left'],
                            means['readout_acc_left'] + sems['readout_acc_left'], color='r', alpha=0.2)
        ax_acc.plot(epochs, means['readout_acc_right'], color='b', label='Right Hemi')
        ax_acc.fill_between(epochs, means['readout_acc_right'] - sems['readout_acc_right'],
                            means['readout_acc_right'] + sems['readout_acc_right'], color='b', alpha=0.2)
        ax_acc.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2)
        ax_acc.set_ylim(0.4, 1.05)
        if col == 0:
            ax_acc.set_ylabel('Readout Acc.')
        if row_acc == 0 and col == 0:
            ax_acc.legend(fontsize=8)
        if row_acc == 0:
            ax_acc.set_title('Corruption@Epoch {}\nReadout Accuracy'.format(corruption_start_epoch))
        else:
            ax_acc.set_title('Corruption@Epoch {}'.format(corruption_start_epoch))

        # Agreement subplot
        ax_agree = axes[row_agree, col]
        ax_agree.plot(epochs, means['agreement_frac'], color='k', label='Empirical')
        ax_agree.fill_between(epochs, means['agreement_frac'] - sems['agreement_frac'],
                            means['agreement_frac'] + sems['agreement_frac'], color='k', alpha=0.2)
        ax_agree.plot(epochs, means['chance_agree'], color='gray', linestyle='--', label='Chance')
        ax_agree.fill_between(epochs, means['chance_agree'] - sems['chance_agree'],
                            means['chance_agree'] + sems['chance_agree'], color='gray', alpha=0.2)
        ax_agree.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2)
        ax_agree.set_ylim(0.4, 1.05)
        if col == 0:
            ax_agree.set_ylabel('L=R Fraction')
        if row_agree == 1 and col == 1:
            ax_agree.legend(fontsize=8)
        if row_agree == 1 and col == 0:
            ax_agree.set_title('L/R Agreement')
        else:
            ax_agree.set_title('')
        # X label only on bottom row
        if row_agree == 3:
            ax_agree.set_xlabel('Epoch')

    # Hide unused axes (bottom row, col=4)
    for row in [2, 3]:
        for col in range(n_cols_bottom, n_cols_top):
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('figs/LR_readoutacc_and_agreement_corrupted_learning_allcorruptionepochs_meansem_4x5.pdf')
    plt.show()


    ## Look at perturbation conditions

    import seaborn as sns
    import glob
    import os

    # Plot the performance at the end of training (last epoch) for all models where corruption started at epoch 7

    corruption_start_epoch = 8
    results_base_dir = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/train_type_modular_corruption'
    pattern = f'corr_epoch_{corruption_start_epoch}_noise_0.70/n_neurons_256_random_seed_*/n_epochs_10_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.20/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'

    # Find all matching result files for this corruption epoch
    search_pattern = os.path.join(results_base_dir, pattern)
    result_files = sorted(glob.glob(search_pattern))

    # Preallocate arrays for each condition (one value per model/seed)
    control_left = []
    leftpert_left = []
    rightpert_left = []

    control_right = []
    leftpert_right = []
    rightpert_right = []

    for file in result_files:
        results_dict = np.load(file, allow_pickle=True)
        # Get the last epoch (post-training) for each condition
        control_left.append(results_dict[-1]['control']['readout_accuracy_left'])
        leftpert_left.append(results_dict[-1]['left_alm_pert']['readout_accuracy_left'])
        rightpert_left.append(results_dict[-1]['right_alm_pert']['readout_accuracy_left'])
        control_right.append(results_dict[-1]['control']['readout_accuracy_right'])
        leftpert_right.append(results_dict[-1]['left_alm_pert']['readout_accuracy_right'])
        rightpert_right.append(results_dict[-1]['right_alm_pert']['readout_accuracy_right'])

    # Convert to numpy arrays
    control_left = np.array(control_left)
    leftpert_left = np.array(leftpert_left)
    rightpert_left = np.array(rightpert_left)

    control_right = np.array(control_right)
    leftpert_right = np.array(leftpert_right)
    rightpert_right = np.array(rightpert_right)

    n_models = len(control_left)

    # Prepare data for plotting
    bar_means_left = [np.mean(control_left), np.mean(leftpert_left), np.mean(rightpert_left)]
    bar_sems_left = [np.std(control_left, ddof=1)/np.sqrt(n_models),
                    np.std(leftpert_left, ddof=1)/np.sqrt(n_models),
                    np.std(rightpert_left, ddof=1)/np.sqrt(n_models)]

    bar_means_right = [np.mean(control_right), np.mean(leftpert_right), np.mean(rightpert_right)]
    bar_sems_right = [np.std(control_right, ddof=1)/np.sqrt(n_models),
                    np.std(leftpert_right, ddof=1)/np.sqrt(n_models),
                    np.std(rightpert_right, ddof=1)/np.sqrt(n_models)]

    labels = ['Control', 'Left Perturb', 'Right Perturb']
    bar_colors = ['#888888', '#e41a1c', '#377eb8']  # grey, red, blue

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # --- Left Hemi subplot ---
    ax = axes[0]
    bars = ax.bar(range(3), bar_means_left, yerr=bar_sems_left, capsize=6, color=bar_colors, alpha=0.7)

    # Scatter the individual model points and connect them
    for i, (c, l, r) in enumerate(zip(control_left, leftpert_left, rightpert_left)):
        ax.plot([0, 1, 2], [c, l, r], color='gray', alpha=0.5, marker='o', markersize=7, linewidth=1.5, zorder=10)

    # Overlay the individual points for each bar
    ax.scatter(np.full(n_models, 0), control_left, color='k', s=40, zorder=20, label='Models')
    ax.scatter(np.full(n_models, 1), leftpert_left, color='k', s=40, zorder=20)
    ax.scatter(np.full(n_models, 2), rightpert_left, color='k', s=40, zorder=20)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Readout Accuracy')
    ax.set_title('Left Hemi Readout Accuracy\n(Corruption@Epoch {})'.format(corruption_start_epoch))
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # --- Right Hemi subplot ---
    ax = axes[1]
    bars = ax.bar(range(3), bar_means_right, yerr=bar_sems_right, capsize=6, color=bar_colors, alpha=0.7)

    for i, (c, l, r) in enumerate(zip(control_right, leftpert_right, rightpert_right)):
        ax.plot([0, 1, 2], [c, l, r], color='gray', alpha=0.5, marker='o', markersize=7, linewidth=1.5, zorder=10)

    ax.scatter(np.full(len(control_right), 0), control_right, color='k', s=40, zorder=20, label='Models')
    ax.scatter(np.full(len(leftpert_right), 1), leftpert_right, color='k', s=40, zorder=20)
    ax.scatter(np.full(len(rightpert_right), 2), rightpert_right, color='k', s=40, zorder=20)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_title('Right Hemi Readout Accuracy\n(Corruption@Epoch {})'.format(corruption_start_epoch))
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()
