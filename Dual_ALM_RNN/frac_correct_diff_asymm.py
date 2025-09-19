"""
Test the performance under perturbation conditions for various asymmetries in the model
"""

import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp
from test_perturbation_eval import test_perturbation_evaluation
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = '42' 

input_asym = [(1,0), (1,0.1), (1,0.2), (1,0.3), (1,0.4), (1,0.5), (1,1), (0.5,1), (0.4,1), (0.3,1), (0.2,1), (0.1,1), (0,1)] # Sane as BK

# Initialize experiment
exp = DualALMRNNExp()
weights_dict = np.load('small_rnn_corrupted_weights_0p5noise_epoch11_40epochs.npy', allow_pickle=True)
weights_dict = np.load('small_rnn_sherlock_weights_corruption_0p7noise_epoch11.npy', allow_pickle=True)
weights_dict = np.load('small_rnn_sherlock_weights_fixed_input.npy', allow_pickle=True)
# weights_dict = np.load('small_rnn_corrupted_weights_fixed_input_0p5noise_epoch15_40epochs.npy', allow_pickle=True)

all_weights = []
control_std = []
for asym in input_asym:


    all_weights += [np.mean([np.sum(np.abs(weights_dict.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict.item()[asym][i][0, exp.n_neurons//2:])) for i in range(10)])]
    control_std += [np.std([np.sum(np.abs(weights_dict.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict.item()[asym][i][0, exp.n_neurons//2:])) for i in range(10)]) / np.sqrt(10)]


    # all_weights += [np.mean([np.sum(np.abs(weights_dict.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict.item()[asym][i][0, :])) for i in range(10)])]
    # control_std += [np.std([np.sum(np.abs(weights_dict.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict.item()[asym][i][0, :])) for i in range(10)]) / np.sqrt(10)]

    # all_weights = [weights_dict.item()[asym][i][0, :exp.n_neurons//2] / weights_dict.item()[asym][i][0, :] for i in range(10)]
    # for seed in range(1,10):
    #     exp = DualALMRNNExp()
    #     exp.configs['xs_left_alm_amp'] = asym[0]
    #     exp.configs['xs_right_alm_amp'] = asym[1]

    #     # For each asymmetry, print the logs path for the config
    #     exp.init_sub_path('train_type_modular')
    #     logs_path = os.path.join(exp.configs['logs_dir'], exp.configs['model_type'], exp.sub_path)
    #     # print(f"Input asymmetry {asym}: logs path = {logs_path}")

    #     weights = np.load(os.path.join(logs_path, 'readout_weights_epoch_final.npy'))
    #     ratio = np.sum(np.abs(weights)[0, :exp.n_neurons//2]) / np.sum(np.abs(weights)[0, :])
    #     # all_weights.append(weights)
    
    # weights_dict[asym] = all_weights
# import pdb; pdb.set_trace()
print(all_weights)
# xlabels=[-1, -0.8, -0.6, -0.2, 0, 0.2, 0.6, 0.8, 1]
xlabels=[-1, -0.9, -0.8, -0.7, -0.6, -0.5, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# plt.plot(xlabels, all_weights, ls='--', marker='o')
plt.errorbar(xlabels, all_weights, yerr=control_std, label='Control', color='black', ls='--')
# plt.errorbar(xlabels, right_pert_acc, yerr=right_pert_std, label='Right pert', color='darkgrey')
# plt.errorbar(xlabels, left_pert_acc, yerr=left_pert_std, label='Left pert', color='lightgrey')
plt.ylabel('Readout weight ratio')
plt.xlabel('Input asymmetry')
# plt.ylim(0, 1)
plt.xticks([-1,0,1],[-1,0,1])
# plt.legend()
plt.savefig('figs/all_competition_fixed_input.pdf')
# plt.savefig('figs/all_competition_tenseeds_corruption_0p7noise_epoch11.pdf')
plt.show()
# breakpoint()

weights_dict_control = np.load('small_rnn_sherlock_weights_epoch40.npy', allow_pickle=True)
weights_dict_control = np.load('small_rnn_sherlock_weights_fixed_input.npy', allow_pickle=True)


all_weights_control, all_weights = [], []
control_std, control_std_control = [], []
for asym in input_asym:

    # all_weights += [np.mean([np.sum(np.abs(weights_dict.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict.item()[asym][i][0, :])) for i in range(10)])]
    # control_std += [np.std([np.sum(np.abs(weights_dict.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict.item()[asym][i][0, :])) for i in range(10)]) / np.sqrt(10)]
    
    # all_weights_control += [np.mean([np.sum(np.abs(weights_dict_control.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict_control.item()[asym][i][0, :])) for i in range(10)])]
    # control_std_control += [np.std([np.sum(np.abs(weights_dict_control.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict_control.item()[asym][i][0, :])) for i in range(10)]) / np.sqrt(10)]

    all_weights += [np.mean([np.sum(np.abs(weights_dict.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict.item()[asym][i][0, exp.n_neurons//2:])) for i in range(10)])]
    control_std += [np.std([np.sum(np.abs(weights_dict.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict.item()[asym][i][0, exp.n_neurons//2:])) for i in range(10)]) / np.sqrt(10)]
    
    all_weights_control += [np.mean([np.sum(np.abs(weights_dict_control.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict_control.item()[asym][i][0, exp.n_neurons//2:])) for i in range(10)])]
    control_std_control += [np.std([np.sum(np.abs(weights_dict_control.item()[asym][i][0, :exp.n_neurons//2])) / np.sum(np.abs(weights_dict_control.item()[asym][i][0, exp.n_neurons//2:])) for i in range(10)]) / np.sqrt(10)]

xlabels=[-1, -0.9, -0.8, -0.7, -0.6, -0.5, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# plt.plot(xlabels, all_weights, ls='--', marker='o')
plt.errorbar(xlabels, all_weights_control, yerr=control_std_control, label='Control', color='black', ls='--')
plt.errorbar(xlabels, all_weights, yerr=control_std, label='Corrupted', color='red', ls='--')
# plt.errorbar(xlabels, right_pert_acc, yerr=right_pert_std, label='Right pert', color='darkgrey')
# plt.errorbar(xlabels, left_pert_acc, yerr=left_pert_std, label='Left pert', color='lightgrey')
plt.ylabel('Readout weight ratio')
plt.xlabel('Input asymmetry')
plt.ylim(0, 1)
plt.title('Corrupted vs Control RNN weights')
plt.xticks([-1,0,1],[-1,0,1])
plt.legend()
# plt.savefig('figs/all_competition_tenseeds_fixed_input_corruption_0p7noise_epoch15vscontrol.pdf')
plt.show()
# breakpoint()


# Plot the readout accuracy per hemipheres for both control and corrupted over asymmetries




# New plot: difference in readout accuracies (Left - Right) on control trials at the final epoch, with error bars across seeds
# Plot both control (train_type_modular_single_readout) and corrupted (train_type_modular_corruption) on the same axes
diff_means = {'control': [], 'corrupted': []}
diff_sems = {'control': [], 'corrupted': []}
control_means = {'left': [], 'right': []}
control_sems = {'left': [], 'right': []}
corrupted_means = {'left': [], 'right': []}
corrupted_sems = {'left': [], 'right': []}
# Using the experiment's logging layout to locate all_val_results_dict.npy for each asymmetry and seed
exp = DualALMRNNExp()
input_asym = [(1.0,0.0), (1.0,0.1), (1.0,0.2), (1.0,0.3), (1.0,0.4), (1.0,0.5), (1.0,1.0), (0.5,1.0), (0.4,1.0), (0.3,1.0), (0.2,1.0), (0.1,1.0), (0.0,1.0)]

for asym in input_asym:
    # compute for both train types
    diffs_per_type = {'control': [], 'corrupted': []}
    control_acc_per_trial_type = {'left': [], 'right': []}
    corrupted_acc_per_trial_type = {'left': [], 'right': []}
    for seed in range(0, 10):
        for label, train_type in [('control', 'train_type_modular_fixed_input'), ('corrupted', 'train_type_modular_corruption_fixed_input')]:
            exp_local = DualALMRNNExp()
            exp_local.configs['xs_left_alm_amp'] = asym[0]
            exp_local.configs['xs_right_alm_amp'] = asym[1]
            exp_local.configs['random_seed'] = seed
            exp_local.configs['train_type'] = train_type
            exp_local.configs['n_epochs'] = 40

            exp_local.init_sub_path(exp_local.configs['train_type'])
            logs_path = os.path.join(exp_local.configs['logs_dir'], exp_local.configs['model_type'], exp_local.sub_path)

            results_path = os.path.join(logs_path, 'all_val_results_dict.npy')
            if not os.path.exists(results_path):
                print(f"Results path {results_path} does not exist")
                continue
            results_dict = np.load(results_path, allow_pickle=True)
            if len(results_dict) == 0:
                print(f"Results dict is empty for {results_path}")
                continue
            final_res = results_dict[-1].item() if isinstance(results_dict[-1], dict) is False else results_dict[-1]
            control = final_res.get('control', {})

            # Prefer explicit control keys if present (single readout), else fall back to readout_accuracy_* keys
            left_acc = control.get('readout_accuracy_left_control', control.get('readout_accuracy_left', np.nan))
            right_acc = control.get('readout_accuracy_right_control', control.get('readout_accuracy_right', np.nan))
            control_acc_per_trial_type['left'].append(left_acc)
            control_acc_per_trial_type['right'].append(right_acc)

            if label == 'corrupted':

                corrupted_acc_per_trial_type['left'].append(left_acc)
                corrupted_acc_per_trial_type['right'].append(right_acc)

            if not np.isnan(left_acc) and not np.isnan(right_acc):
                diffs_per_type[label].append(left_acc - right_acc)

    # Aggregate across seeds for this asymmetry
    for label in ['control', 'corrupted']:
        vals = diffs_per_type[label]
        diff_means[label].append(np.mean(vals) if len(vals) else np.nan)
        diff_sems[label].append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan)

    for label in ['left', 'right']:
        vals = control_acc_per_trial_type[label]
        control_means[label].append(np.mean(vals) if len(vals) else np.nan)
        control_sems[label].append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan)
        vals = corrupted_acc_per_trial_type[label]
        corrupted_means[label].append(np.mean(vals) if len(vals) else np.nan)
        corrupted_sems[label].append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan)

# Map asymmetries to the same xlabels used above
xlabels=[-1, -0.9, -0.8, -0.7, -0.6, -0.5, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

plt.figure(figsize=(8,5))
plt.errorbar(xlabels, diff_means['control'], yerr=diff_sems['control'], label='Left - Right (control)', color='black', ls='-')
plt.errorbar(xlabels, diff_means['corrupted'], yerr=diff_sems['corrupted'], label='Left - Right (corrupted)', color='red', ls='-')
plt.xlabel('Input asymmetry')
plt.ylabel('Readout accuracy difference (Left - Right, control, final epoch)')
plt.xticks([-1,0,1],[-1,0,1])
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.legend()
os.makedirs('figs', exist_ok=True)
plt.savefig('figs/readout_control_acc_diff_by_hemi_final_epoch_fixed_input.pdf', bbox_inches='tight')
# plt.savefig('figs/readout_control_acc_diff_by_hemi_final_epoch.png', dpi=200, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8,5))
plt.errorbar(xlabels, control_means['left'], yerr=control_sems['left'], label='Left (control)', color='red', ls='-', linewidth=2)
plt.errorbar(xlabels, control_means['right'], yerr=control_sems['right'], label='Right (control)', color='blue', ls='-', linewidth=2)
plt.errorbar(xlabels, corrupted_means['left'], yerr=corrupted_sems['left'], label='Left (corrupted)', color='pink', ls='-', linewidth=2)
plt.errorbar(xlabels, corrupted_means['right'], yerr=corrupted_sems['right'], label='Right (corrupted)', color='lightblue', ls='-', linewidth=2)
plt.xlabel('Input asymmetry')
plt.ylabel('Readout accuracy')
plt.xticks([-1,0,1],[-1,0,1])
plt.legend()
plt.savefig('figs/readout_acc_by_hemi_final_epoch_fixed_input.pdf', bbox_inches='tight')
plt.show()



if False:
    # input_asym = [(0, 1)] # Sane as BK

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

    input_asym = [(1.0,0.0), (1.0,0.1), (1.0,0.2), (1.0,0.5), (1.0,1.0), (0.5,1.0), (0.2,1.0), (0.1,1.0), (0.0,1.0)] # Same as BK

    for input_asym in input_asym:
        temp_control_acc = []
        temp_right_pert_acc = []
        temp_left_pert_acc = []
        for seed in range(5,10):
            path = '/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh/train_type_modular/onehot/n_neurons_256_random_seed_{}/n_epochs_10_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/'.format(seed, input_asym[0], input_asym[1])
            results_dict = np.load(os.path.join(path, 'all_val_results_dict.npy'), allow_pickle=True)
            temp_control_acc += [np.mean([results_dict[-1]['control']['readout_accuracy_left'], results_dict[-1]['control']['readout_accuracy_right']])]
            temp_right_pert_acc += [np.mean([results_dict[-1]['right_alm_pert']['readout_accuracy_left'], results_dict[-1]['right_alm_pert']['readout_accuracy_right']])]
            temp_left_pert_acc += [np.mean([results_dict[-1]['left_alm_pert']['readout_accuracy_left'], results_dict[-1]['left_alm_pert']['readout_accuracy_right']])]
        
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
    plt.savefig('figs/frac_correct_diff_asymm_onehot.pdf')
    plt.show()


