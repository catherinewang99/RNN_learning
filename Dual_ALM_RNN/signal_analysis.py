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


# Plot d prime as a heatmap as a function of input asymmetry and corruption noise

def get_d_prime(mean1, mean2, std1, std2):
    return (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)

if False:

    # noise_sd = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    noise_sd = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    input_asym = [(1.0,0.0), (1.0,0.1), (1.0,0.2), (1.0,0.3), (1.0,0.4), (1.0,0.5), (1.0,1.0), (0.5,1.0), (0.4,1.0), (0.3,1.0), (0.2,1.0), (0.1,1.0), (0.0,1.0)] # Same as BK

    signal_mean = 0.5
    all_dprimes = []
    for noise in noise_sd:
        noise_dprimes = []
        for asym in input_asym:
            left_ratio, right_ratio = asym

            left_sd1 = np.sqrt((0.05 * left_ratio)**2 + noise**2)
            right_sd1 = np.sqrt((0.05 * right_ratio)**2 + 0.1**2)

            left_sd2 = np.sqrt(0 + noise**2) # channel has mean, sd = 0 
            right_sd2 = np.sqrt(0 + 0.1**2)

            left_dprime = get_d_prime(signal_mean*left_ratio, 0, left_sd1, left_sd2)
            right_dprime = get_d_prime(signal_mean*right_ratio, 0, right_sd1, right_sd2)

            noise_dprimes.append(left_dprime - right_dprime)
        all_dprimes.append(noise_dprimes)




    plt.imshow(all_dprimes, cmap='RdBu_r')
    plt.xlabel('Input asymmetry')
    plt.xticks([0, 6, 12], [-1, 0, 1])
    plt.ylabel('Corruption noise (SD)')
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    plt.title('D prime (Left - Right)')
    plt.colorbar()
    plt.savefig('figs/d_prime_heatmap.pdf')
    plt.show()





    # Gradient of d prime as a function of asymmetry and signal size

    signal_means = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    input_asym = [(1.0,0.0), (1.0,0.1), (1.0,0.2), (1.0,0.3), (1.0,0.4), (1.0,0.5), (1.0,1.0), (0.5,1.0), (0.4,1.0), (0.3,1.0), (0.2,1.0), (0.1,1.0), (0.0,1.0)] # Same as BK

    all_dprimes = []
    for signal_mean in signal_means:
        noise_dprimes = []
        for asym in input_asym:
            left_ratio, right_ratio = asym

            left_sd1 = np.sqrt((0.05 * left_ratio)**2 + 0.1**2)
            right_sd1 = np.sqrt((0.05 * right_ratio)**2 + 0.1**2)

            left_sd2 = np.sqrt(0 + 0.1**2) # channel has mean, sd = 0 
            right_sd2 = np.sqrt(0 + 0.1**2)

            left_dprime = get_d_prime(signal_mean*left_ratio, 0, left_sd1, left_sd2)
            right_dprime = get_d_prime(signal_mean*right_ratio, 0, right_sd1, right_sd2)

            noise_dprimes.append(left_dprime - right_dprime)
        all_dprimes.append(noise_dprimes)




    plt.imshow(all_dprimes, cmap='RdBu_r')
    plt.xlabel('Input asymmetry')
    plt.xticks([0, 6, 12], [-1, 0, 1])
    plt.ylabel('Signal mean')
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    plt.title('D prime (Left - Right)')
    plt.colorbar()
    plt.savefig('figs/d_prime_heatmap_signal_mean.pdf')
    plt.show()


# delta of d prime vs readout accuracy during corruption case
input_asym = [(1.0,0.0), (1.0,0.1), (1.0,0.2), (1.0,0.3), (1.0,0.4), (1.0,0.5), (1.0,1.0), (0.5,1.0), (0.4,1.0), (0.3,1.0), (0.2,1.0), (0.1,1.0), (0.0,1.0)] # Same as BK

exp = DualALMRNNExp()
exp.configs['train_type'] = 'train_type_modular_corruption'
exp.configs['corruption_type'] = 'gaussian'
exp.configs['corruption_start_epoch'] = 11
exp.configs['corruption_noise'] = 0.5
exp.configs['n_epochs'] = 40
signal_mean, noise = 0.5, 0.5

# Collect data across all seeds for each asymmetry
all_dprimes_per_asym = []
all_acc_per_asym = []
all_readout_ratios_per_asym = []

for left_ratio, right_ratio in input_asym:
    dprimes_for_asym = []
    accs_for_asym = []
    readout_ratios_for_asym = []
    
    for seed in range(0, 10):
        exp_local = DualALMRNNExp()
        exp_local.configs['train_type'] = 'train_type_modular_corruption'
        exp_local.configs['corruption_type'] = 'gaussian'
        exp_local.configs['corruption_start_epoch'] = 11
        exp_local.configs['corruption_noise'] = 0.5
        exp_local.configs['n_epochs'] = 40
        exp_local.configs['xs_left_alm_amp'] = left_ratio
        exp_local.configs['xs_right_alm_amp'] = right_ratio
        exp_local.configs['random_seed'] = seed
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
        # corrupted = final_res.get('corrupted', {})
        left_acc = control.get('readout_accuracy_left_control', control.get('readout_accuracy_left', np.nan))
        right_acc = control.get('readout_accuracy_right_control', control.get('readout_accuracy_right', np.nan))

        # Load readout weights
        weights_path = os.path.join(logs_path, 'readout_weights_epoch_final.npy')
        if os.path.exists(weights_path):
            weights = np.load(weights_path)
            # Calculate readout weight ratio (left hemisphere / right hemisphere)
            # Assuming weights shape is (n_neurons,) where first half is left, second half is right
            n_neurons = weights.shape[0]
            left_weights = weights[:n_neurons//2]
            right_weights = weights[n_neurons//2:]
            
            # Calculate ratio of mean absolute weights
            left_weight_magnitude = np.mean(np.abs(left_weights))
            right_weight_magnitude = np.mean(np.abs(right_weights))
            
            if right_weight_magnitude > 0:  # Avoid division by zero
                readout_ratio = left_weight_magnitude / right_weight_magnitude
            else:
                readout_ratio = np.nan
        else:
            print(f"Weights path {weights_path} does not exist")
            readout_ratio = np.nan

        if not np.isnan(left_acc) and not np.isnan(right_acc) and not np.isnan(readout_ratio):
            left_sd1 = np.sqrt((0.05 * left_ratio)**2 + noise**2)
            right_sd1 = np.sqrt((0.05 * right_ratio)**2 + 0.1**2)

            left_sd2 = np.sqrt(0 + 0.1**2) # channel has mean, sd = 0 
            right_sd2 = np.sqrt(0 + 0.1**2)

            left_dprime = get_d_prime(signal_mean*left_ratio, 0, left_sd1, left_sd2)
            right_dprime = get_d_prime(signal_mean*right_ratio, 0, right_sd1, right_sd2)
            
            dprimes_for_asym.append(left_dprime - right_dprime)
            accs_for_asym.append(left_acc - right_acc)
            readout_ratios_for_asym.append(readout_ratio)
    
    all_dprimes_per_asym.append(dprimes_for_asym)
    all_acc_per_asym.append(accs_for_asym)
    all_readout_ratios_per_asym.append(readout_ratios_for_asym)
    
    # Debug: print data availability for this asymmetry
    print(f"Asymmetry ({left_ratio:.1f}, {right_ratio:.1f}): {len(dprimes_for_asym)} valid seeds out of 10")


# Calculate means and standard errors across seeds for each asymmetry
# Handle empty arrays to avoid warnings
dprime_means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in all_dprimes_per_asym]
dprime_sems = [np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0 for vals in all_dprimes_per_asym]
acc_means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in all_acc_per_asym]
acc_sems = [np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0 for vals in all_acc_per_asym]
readout_ratio_means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in all_readout_ratios_per_asym]
readout_ratio_sems = [np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0 for vals in all_readout_ratios_per_asym]

# Plot 1: D-prime vs readout accuracy
plt.figure(figsize=(8, 6))
plt.errorbar(dprime_means, acc_means, xerr=dprime_sems, yerr=acc_sems, 
             fmt='o', capsize=5, capthick=2, markersize=6)
plt.xlabel('D prime (Left - Right)')
plt.ylabel('Readout accuracy difference (Left - Right)')
plt.grid(True, alpha=0.3)
plt.title('D-prime vs Readout Accuracy')
os.makedirs('figs', exist_ok=True)
plt.savefig('figs/d_prime_vs_readout_accuracy.pdf', bbox_inches='tight')
plt.show()

# Plot 2: D-prime vs readout weight ratio
plt.figure(figsize=(8, 6))
plt.errorbar(dprime_means, readout_ratio_means, xerr=dprime_sems, yerr=readout_ratio_sems, 
             fmt='s', capsize=5, capthick=2, markersize=6, color='red')
plt.xlabel('D prime (Left - Right)')
plt.ylabel('Readout weight ratio (Left / Right)')
plt.grid(True, alpha=0.3)
plt.title('D-prime vs Readout Weight Ratio')
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal weights')
plt.legend()
plt.savefig('figs/d_prime_vs_readout_weight_ratio.pdf', bbox_inches='tight')
plt.show()

# Plot 3: Readout weight ratios across asymmetries
plt.figure(figsize=(10, 6))
xlabels = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.errorbar(xlabels, readout_ratio_means, yerr=readout_ratio_sems, 
             fmt='o', capsize=5, capthick=2, markersize=6, color='green')
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal weights')
plt.xlabel('Input asymmetry')
plt.ylabel('Readout weight ratio (Left / Right)')
plt.title('Readout Weight Ratio vs Input Asymmetry')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('figs/readout_weight_ratio_vs_asymmetry.pdf', bbox_inches='tight')
plt.show()
