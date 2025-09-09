"""
Visualize the readout accuracies and agreement fractions of the small RNN for different input asymmetries
"""

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


### Plot single examples, corrupted vs control trials###

if False:
    exp = DualALMRNNExp()

    if exp.configs['train_type'] == 'train_type_modular_corruption':
        results_dict = np.load(
            'dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_corruption/onehot_cor_type_{}_epoch_{}_noise_{:.2f}/n_neurons_4_random_seed_{}/n_epochs_{}_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(
                exp.configs['corruption_type'],
                exp.configs['corruption_start_epoch'],
                float(exp.configs['corruption_noise']),
                exp.configs['random_seed'],
                exp.configs['n_epochs'],
                float(exp.configs['xs_left_alm_amp']),
                float(exp.configs['xs_right_alm_amp'])
            ),
            allow_pickle=True
        )

    epochs = np.arange(1, len(results_dict) + 1)
    readout_acc_left = np.array([results_dict[i]['control']['readout_accuracy_left_control'] for i in range(len(results_dict))])
    readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right_control'] for i in range(len(results_dict))])
    readout_acc_left_corrupted = np.array([results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))])
    readout_acc_right_corrupted = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])


    corruption_start_epoch = exp.configs['corruption_start_epoch']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Top subplot: Readout accuracies in control trials
    ax1.plot(epochs, readout_acc_left, color='r', label='Left Hemi Readout Accuracy')
    ax1.plot(epochs, readout_acc_right, color='b', label='Right Hemi Readout Accuracy')
    ax1.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2, label='Corruption Start (Epoch {})'.format(corruption_start_epoch))
    ax1.set_ylabel('Readout Accuracy (Control)')
    ax1.set_title('Readout Accuracy Over Training (Control Trials)')
    ax1.set_xticks(epochs)
    ax1.set_ylim(0.4, 1.05)
    ax1.legend()

    # Bottom subplot: readout accuracies in corrupted trials
    ax2.plot(epochs, readout_acc_left_corrupted, color='r', label='Left Hemi Readout Accuracy')
    ax2.plot(epochs, readout_acc_right_corrupted, color='b', label='Right Hemi Readout Accuracy')
    ax2.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2, label='Corruption Start (Epoch {})'.format(corruption_start_epoch))
    ax2.set_ylabel('Readout Accuracy (Corrupted)')
    ax2.set_title('Readout Accuracy Over Training (Corrupted Trials)')
    ax2.set_xticks(epochs)
    ax2.set_ylim(0.4, 1.05)
    ax2.legend()


    plt.tight_layout()
    plt.savefig('figs/LR_readoutacc_ctl_vs_corr_learning_{}_L{}_R{}_epoch_{}_noise_{}0_type_{}.pdf'.format(exp.configs['train_type'], 
                                                                                                            exp.configs['xs_left_alm_amp'], 
                                                                                                            exp.configs['xs_right_alm_amp'], 
                                                                                                            exp.configs['corruption_start_epoch'], 
                                                                                                            exp.configs['corruption_noise'], 
                                                                                                            exp.configs['corruption_type']))
    plt.show()
### Plot single examples, corrupted vs control and agreement###

if False:
    exp = DualALMRNNExp()

    if exp.configs['train_type'] == 'train_type_modular_corruption':
        results_dict = np.load(
            'dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_corruption/onehot_cor_type_{}_epoch_{}_noise_{:.2f}/n_neurons_4_random_seed_{}/n_epochs_30_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(
                exp.configs['corruption_type'],
                exp.configs['corruption_start_epoch'],
                float(exp.configs['corruption_noise']),
                exp.configs['random_seed'],
                float(exp.configs['xs_left_alm_amp']),
                float(exp.configs['xs_right_alm_amp'])
            ),
            allow_pickle=True
        )
    else:
        results_dict = np.load(
            'dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_single_readout/n_neurons_4_random_seed_{}/n_epochs_30_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(
                exp.configs['random_seed'],
                float(exp.configs['xs_left_alm_amp']),
                float(exp.configs['xs_right_alm_amp'])
            ),
            allow_pickle=True
        )
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
    if exp.configs['train_type'] == 'train_type_modular_corruption':
        ax1.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2, label='Corruption Start (Epoch {})'.format(corruption_start_epoch))
    ax1.set_ylabel('Readout Accuracy (Control)')
    ax1.set_title('Readout Accuracy Over Training (Control Trials)')
    ax1.set_xticks(epochs)
    ax1.set_ylim(0.4, 1.05)
    ax1.legend()

    # Bottom subplot: Agreement
    ax2.plot(epochs, agreement_frac, color='k', label='Empirical Agreement (L=R)')
    ax2.plot(epochs, chance_agree, color='gray', linestyle='--', label='Chance Level (Binomial)')
    if exp.configs['train_type'] == 'train_type_modular_corruption':
        ax2.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2, label='Corruption Start (Epoch {})'.format(corruption_start_epoch))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Fraction of Trials with L=R')
    ax2.set_title('Left/Right ALM Agreement Over Training (Control Trials)')
    ax2.set_xticks(epochs)
    ax2.set_ylim(0.4, 1.05)
    ax2.legend()

    plt.tight_layout()
    if exp.configs['train_type'] == 'train_type_modular_corruption':
        plt.savefig('figs/LR_readoutacc_and_agreement_learning_{}_L{}_R{}_epoch_{}_noise_{}0_type_{}.pdf'.format(exp.configs['train_type'], 
                                                                                                                exp.configs['xs_left_alm_amp'], 
                                                                                                                exp.configs['xs_right_alm_amp'], 
                                                                                                                exp.configs['corruption_start_epoch'], 
                                                                                                                exp.configs['corruption_noise'], 
                                                                                                                exp.configs['corruption_type']))
    else:
        plt.savefig('figs/LR_readoutacc_and_agreement_learning_{}_L{}_R{}_type_{}.pdf'.format(exp.configs['train_type'], 
                                                                                                exp.configs['xs_left_alm_amp'], 
                                                                                                exp.configs['xs_right_alm_amp'], 
                                                                                                exp.configs['train_type']))
    plt.show()





# Plot all results for all seeds
if True:


    ### Plot control results over seeds ###

    exp = DualALMRNNExp()
    input_asym = [(1.0,0.0), (1.0,0.1), (1.0,0.2), (1.0,0.3), (1.0,0.4), (1.0,0.5), (1.0,1.0), (0.5,1.0), (0.4,1.0), (0.3,1.0), (0.2,1.0), (0.1,1.0), (0.0,1.0)] # Same as BK
    input_asym = [(1.0,0.0), (1.0,0.1), (1.0,0.2), (1.0,0.3), (1.0,0.4), (1.0,0.5), (1.0,1.0), (0.5,1.0), (0.4,1.0), (0.3,1.0), (0.2,1.0), (0.1,1.0), (0.0,1.0)] # Sane as BK

    # input_asym = [(1.0,1.0)]

    # Define the range of random seeds to analyze
    random_seeds = range(10)  # 0 to 9

    # Create figs directory if it doesn't exist
    os.makedirs('figs', exist_ok=True)

    # Loop through all input asymmetries
    for left_amp, right_amp in input_asym:
        print(f"\n{'='*60}")
        print(f"Processing Input Asymmetry: Left={left_amp}, Right={right_amp}")
        print(f"{'='*60}")
        
        # Update experiment configs for this asymmetry
        exp.configs['xs_left_alm_amp'] = left_amp
        exp.configs['xs_right_alm_amp'] = right_amp
        
        # Initialize arrays to store results across all seeds for this asymmetry
        all_readout_acc_left = []
        all_readout_acc_right = []
        all_agreement_frac = []
        all_chance_agree = []
        
        # Load results for each random seed
        for seed in random_seeds:
            print(f"Loading results for random seed {seed}...")
            
            # Update the experiment configs to use the current seed
            exp.configs['random_seed'] = seed

            try:
                if exp.configs['train_type'] == 'train_type_modular_corruption':
                    results_dict = np.load('/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_corruption/onehot_cor_type_{}_epoch_{}_noise_{}0/n_neurons_4_random_seed_{}/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(exp.configs['corruption_type'], exp.configs['corruption_start_epoch'], exp.configs['corruption_noise'], seed, left_amp, right_amp), allow_pickle=True)
                elif exp.configs['train_type'] == 'train_type_modular_single_readout':
                    results_dict = np.load('/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_single_readout/n_neurons_4_random_seed_{}/n_epochs_40_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(seed, left_amp, right_amp), allow_pickle=True)
                elif exp.configs['train_type'] == 'train_type_modular_symmetric':
                    results_dict = np.load('/Users/catherinewang/Desktop/RNN/Dual_ALM_RNN/dual_alm_rnn_logs/TwoHemiRNNTanh_single_readout/train_type_modular_symmetric/n_neurons_4_random_seed_{}/n_epochs_30_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{}0_right_alm_amp_{}0/init_cross_hemi_rel_factor_0.20/all_val_results_dict.npy'.format(seed, left_amp, right_amp), allow_pickle=True)
                # Extract metrics for this seed
                readout_acc_left = np.array([results_dict[i]['control']['readout_accuracy_left'] for i in range(len(results_dict))])
                readout_acc_right = np.array([results_dict[i]['control']['readout_accuracy_right'] for i in range(len(results_dict))])
                
                n_trials_agreed = np.array([results_dict[i]['control']['n_trials_agreed'] for i in range(len(results_dict))])
                n_trials = np.array([results_dict[i]['control']['n_trials'] for i in range(len(results_dict))])
                agreement_frac = n_trials_agreed / n_trials
                
                # Compute chance level: p(agree) = p_L^2 + (1-p_L)^2 if p_L = p_R, but here use both
                chance_agree = readout_acc_left * readout_acc_right + (1 - readout_acc_left) * (1 - readout_acc_right)
                
                # Store results
                all_readout_acc_left.append(readout_acc_left)
                all_readout_acc_right.append(readout_acc_right)
                all_agreement_frac.append(agreement_frac)
                all_chance_agree.append(chance_agree)
                
            except FileNotFoundError:
                print(f"Warning: Results not found for random seed {seed}, skipping...")
                continue
        
        # Skip if no data was loaded for this asymmetry
        if len(all_readout_acc_left) == 0:
            print(f"No data found for asymmetry Left={left_amp}, Right={right_amp}, skipping...")
            continue
        
        # Convert to numpy arrays for easier computation
        all_readout_acc_left = np.array(all_readout_acc_left)  # Shape: (n_seeds, n_epochs)
        all_readout_acc_right = np.array(all_readout_acc_right)
        all_agreement_frac = np.array(all_agreement_frac)
        all_chance_agree = np.array(all_chance_agree)
        
        # Calculate mean and standard deviation across seeds
        mean_readout_acc_left = np.mean(all_readout_acc_left, axis=0)
        std_readout_acc_left = np.std(all_readout_acc_left, axis=0) / np.sqrt(len(all_readout_acc_left))
        mean_readout_acc_right = np.mean(all_readout_acc_right, axis=0)
        std_readout_acc_right = np.std(all_readout_acc_right, axis=0) / np.sqrt(len(all_readout_acc_right))
        
        mean_agreement_frac = np.mean(all_agreement_frac, axis=0)
        std_agreement_frac = np.std(all_agreement_frac, axis=0) / np.sqrt(len(all_agreement_frac))
        mean_chance_agree = np.mean(all_chance_agree, axis=0)
        std_chance_agree = np.std(all_chance_agree, axis=0) / np.sqrt(len(all_chance_agree))
        
        # Get epochs (assuming all seeds have the same number of epochs)
        epochs = np.arange(1, all_readout_acc_left.shape[1] + 1)
        
        if exp.configs['train_type'] == 'train_type_modular_corruption':
            corruption_start_epoch = exp.configs['corruption_start_epoch']
        
        # Create figure for this asymmetry
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        
        # Add overall title for the figure
        fig.suptitle(f'Input Asymmetry: Left={left_amp}, Right={right_amp}', fontsize=14, fontweight='bold')
        
        # Top subplot: Readout accuracies with shaded error margins
        ax1.plot(epochs, mean_readout_acc_left, color='r', label='Left Hemi Readout Accuracy', linewidth=2)
        ax1.fill_between(epochs, 
                        mean_readout_acc_left - std_readout_acc_left, 
                        mean_readout_acc_left + std_readout_acc_left, 
                        color='r', alpha=0.3)
        
        ax1.plot(epochs, mean_readout_acc_right, color='b', label='Right Hemi Readout Accuracy', linewidth=2)
        ax1.fill_between(epochs, 
                        mean_readout_acc_right - std_readout_acc_right, 
                        mean_readout_acc_right + std_readout_acc_right, 
                        color='b', alpha=0.3)
        
        if exp.configs['train_type'] == 'train_type_modular_corruption':
            ax1.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2, label='Corruption Start (Epoch {})'.format(corruption_start_epoch))
        ax1.set_ylabel('Readout Accuracy (Control)')
        ax1.set_title('Readout Accuracy Over Training (Control Trials) - Mean ± SD across {} seeds'.format(len(all_readout_acc_left)))
        ax1.set_xticks(epochs)
        ax1.set_ylim(0.4, 1.05)
        ax1.legend()
        
        # Bottom subplot: Agreement with shaded error margins
        ax2.plot(epochs, mean_agreement_frac, color='k', label='Empirical Agreement (L=R)', linewidth=2)
        ax2.fill_between(epochs, 
                        mean_agreement_frac - std_agreement_frac, 
                        mean_agreement_frac + std_agreement_frac, 
                        color='k', alpha=0.3)
        
        ax2.plot(epochs, mean_chance_agree, color='gray', linestyle='--', label='Chance Level (Binomial)', linewidth=2)
        ax2.fill_between(epochs, 
                        mean_chance_agree - std_chance_agree, 
                        mean_chance_agree + std_chance_agree, 
                        color='gray', alpha=0.2)
        
        if exp.configs['train_type'] == 'train_type_modular_corruption':
            ax2.axvline(corruption_start_epoch, color='r', linestyle=':', linewidth=2, label='Corruption Start (Epoch {})'.format(corruption_start_epoch))
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Fraction of Trials with L=R')
        ax2.set_title('Left/Right ALM Agreement Over Training (Control Trials) - Mean ± SD across {} seeds'.format(len(all_readout_acc_left)))
        ax2.set_xticks(epochs)
        ax2.set_ylim(0.4, 1.05)
        ax2.legend()
        
        plt.tight_layout()
        
        # Create filename based on asymmetry
        filename = f'asymmetry_L{left_amp}_R{right_amp}'
        if exp.configs['train_type'] == 'train_type_modular_corruption':
            filename += f'_corruption_{exp.configs["corruption_type"]}_epoch_{exp.configs["corruption_start_epoch"]}_noise_{exp.configs["corruption_noise"]}'
        
        # Save the figure
        save_path = os.path.join('figs', f'{filename}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
        
        # Also save as PDF for high quality
        save_path_pdf = os.path.join('figs', f'{filename}.pdf')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        print(f"Saved figure: {save_path_pdf}")
        
        plt.close()  # Close the figure to free memory
        
        # Print summary statistics for this asymmetry
        print(f"\nSummary for Left={left_amp}, Right={right_amp} across {len(all_readout_acc_left)} random seeds:")
        print(f"Final Left Readout Accuracy: {mean_readout_acc_left[-1]:.3f} ± {std_readout_acc_left[-1]:.3f}")
        print(f"Final Right Readout Accuracy: {mean_readout_acc_right[-1]:.3f} ± {std_readout_acc_right[-1]:.3f}")
        print(f"Final Agreement Fraction: {mean_agreement_frac[-1]:.3f} ± {std_agreement_frac[-1]:.3f}")
        print(f"Final Chance Agreement: {mean_chance_agree[-1]:.3f} ± {std_chance_agree[-1]:.3f}")

    print(f"\n{'='*60}")
    print("All figures saved to the 'figs/' directory!")
    print(f"{'='*60}")


# Plot asymmetry vs readout accuracy
