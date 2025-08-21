#!/usr/bin/env python3
"""
Minimal 2-unit Dual ALM RNN Analysis
Analyzes how sensory input strength asymmetries affect readout weights and cross-hemisphere connections
Uses prebuilt functions from dual_alm_rnn_exp.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt
import time
from dual_alm_rnn_exp import DualALMRNNExp
import dual_alm_rnn_models

# Input asymmetry combinations to test
input_asym = [(1.0,0.0), (1.0,0.1), (1.0,0.2), (1.0,0.5), (1.0,1.0), 
              (0.5,1.0), (0.2,1.0), (0.1,1.0), (0.0,1.0)]  # Same as BK

def train_minimal_rnn(configs, left_amp, right_amp, exp):
    """Train a minimal RNN with specific input amplitudes using prebuilt functions"""
    print(f"\nTraining RNN with left_amp={left_amp:.1f}, right_amp={right_amp:.1f}")
    
    # Modify configs for minimal RNN
    configs['n_neurons'] = 2  # Only 2 units total
    configs['n_epochs'] = 10  # Train for 10 epochs
    configs['one_hot'] = 1    # Use one-hot encoding
    
    # Update configs for this specific training run
    configs['xs_left_alm_amp'] = left_amp
    configs['xs_right_alm_amp'] = right_amp
    
    # Initialize model
    model_type = configs['model_type']
    model = getattr(dual_alm_rnn_models, model_type)(configs, exp.a, exp.pert_begin, exp.pert_end)
    
    # Set device
    device = torch.device("cpu")  # Use CPU to avoid MPS issues
    model = model.to(device)
    
    # Load data
    train_save_path = os.path.join(exp.configs['data_dir'], 'train')
    train_sensory_inputs = np.load(os.path.join(train_save_path, 'onehot_sensory_inputs.npy'))
    train_trial_type_labels = np.load(os.path.join(train_save_path, 'onehot_trial_type_labels.npy'))
    
    val_save_path = os.path.join(exp.configs['data_dir'], 'val')
    val_sensory_inputs = np.load(os.path.join(val_save_path, 'onehot_sensory_inputs.npy'))
    val_trial_type_labels = np.load(os.path.join(val_save_path, 'onehot_trial_type_labels.npy'))
    
    # Create data loaders
    train_set = data.TensorDataset(torch.tensor(train_sensory_inputs), torch.tensor(train_trial_type_labels))
    train_loader = data.DataLoader(train_set, batch_size=configs['bs'], shuffle=True, drop_last=True)
    
    val_set = data.TensorDataset(torch.tensor(val_sensory_inputs), torch.tensor(val_trial_type_labels))
    val_loader = data.DataLoader(val_set, batch_size=configs['bs'], shuffle=False)
    
    # Explicitly include readout weights
    trainable_params = []
    for name, param in model.named_parameters():
        if 'rnn_cell' in name or 'readout_linear' in name:
            trainable_params.append(param)

    # Initialize optimizer and loss
    optimizer = optim.Adam(trainable_params, lr=configs['lr'], weight_decay=configs['l2_weight_decay'])
    # optimizer = optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['l2_weight_decay'])
    loss_fct = nn.BCEWithLogitsLoss()
    
    # Training loop using prebuilt functions
    all_epoch_train_losses = []
    all_epoch_train_scores = []
    all_epoch_val_losses = []
    all_epoch_val_scores = []
    
    for epoch in range(configs['n_epochs']):
        print(f'\nEpoch {epoch+1}/{configs["n_epochs"]}')
        
        # Use prebuilt train_helper function
        train_losses, train_scores = exp.train_helper(model, device, train_loader, optimizer, epoch, loss_fct)
        
        # Use prebuilt val_helper function
        val_loss, val_score = exp.val_helper(model, device, val_loader, loss_fct)
        
        # Convert tensors to numpy safely - detach gradients first
        if isinstance(train_losses[0], torch.Tensor):
            train_losses_np = [loss.detach().cpu().numpy() if loss.requires_grad else loss.cpu().numpy() for loss in train_losses]
        else:
            train_losses_np = train_losses
            
        if isinstance(train_scores[0], torch.Tensor):
            train_scores_np = [score.detach().cpu().numpy() if score.requires_grad else score.cpu().numpy() for score in train_scores]
        else:
            train_scores_np = train_scores
        
        # Record metrics
        all_epoch_train_losses.extend(train_losses_np)
        all_epoch_train_scores.extend(train_scores_np)
        all_epoch_val_losses.append(val_loss)
        all_epoch_val_scores.append(val_score)
        
        print(f'Epoch {epoch+1} Summary: Train Loss: {np.mean(train_losses_np):.4f}, '
              f'Train Acc: {np.mean(train_scores_np)*100:.1f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_score*100:.1f}%')
    
    # Extract final weights
    final_weights = {}
    for name, param in model.named_parameters():
        final_weights[name] = param.data.cpu().numpy().copy()
    
    return final_weights, all_epoch_train_losses, all_epoch_val_losses, all_epoch_train_scores, all_epoch_val_scores



def brute_force_train_minimal_rnn(configs, left_amp, right_amp, exp):
    """Brute force train a minimal RNN with specific input amplitudes using prebuilt functions"""
    print(f"\nTraining RNN with left_amp={left_amp:.1f}, right_amp={right_amp:.1f}")
    
    # Modify configs for minimal RNN
    configs['n_neurons'] = 2  # Only 2 units total
    configs['n_epochs'] = 10  # Train for 10 epochs
    configs['one_hot'] = 1    # Use one-hot encoding
    
    # Update configs for this specific training run
    configs['xs_left_alm_amp'] = left_amp
    configs['xs_right_alm_amp'] = right_amp
    
    # Initialize model
    model_type = configs['model_type']
    model = getattr(dual_alm_rnn_models, model_type)(configs, exp.a, exp.pert_begin, exp.pert_end)
    
    # Set device
    device = torch.device("cpu")  # Use CPU to avoid MPS issues
    model = model.to(device)
    
    # Load data
    train_save_path = os.path.join(exp.configs['data_dir'], 'train')
    train_sensory_inputs = np.load(os.path.join(train_save_path, 'onehot_sensory_inputs.npy'))
    train_trial_type_labels = np.load(os.path.join(train_save_path, 'onehot_trial_type_labels.npy'))
    
    val_save_path = os.path.join(exp.configs['data_dir'], 'val')
    val_sensory_inputs = np.load(os.path.join(val_save_path, 'onehot_sensory_inputs.npy'))
    val_trial_type_labels = np.load(os.path.join(val_save_path, 'onehot_trial_type_labels.npy'))
    
    # Create data loaders
    train_set = data.TensorDataset(torch.tensor(train_sensory_inputs), torch.tensor(train_trial_type_labels))
    train_loader = data.DataLoader(train_set, batch_size=configs['bs'], shuffle=True, drop_last=True)
    
    val_set = data.TensorDataset(torch.tensor(val_sensory_inputs), torch.tensor(val_trial_type_labels))
    val_loader = data.DataLoader(val_set, batch_size=configs['bs'], shuffle=False)
    
    grid = np.linspace(0.0, 1.0, 50)
    best_loss = float('inf')
    best_w = None

    for W_l_readout, W_r_readout, W_l_r, W_r_r, W_l_r, W_r_l in itertools.product(grid, repeat=6):




def analyze_weight_relationships(configs, weights_dict, left_amp, right_amp):
    """Analyze the weight relationships of interest"""
    results = {}
    
    if configs['model_type'] == 'TwoHemiRNNTanh_single_readout':
        results['readout_all'] = weights_dict['readout_linear.weight'] # Left unit to left output
        num_neurons = 2
        results['output_ratio'] = np.abs(weights_dict['readout_linear.weight'][0,:num_neurons//2] / weights_dict['readout_linear.weight'][0,num_neurons//2:]) # Left output : right output
        print(results['output_ratio'])
    else:
        # Readout weights for each hemisphere
        results['readout_left'] = weights_dict['readout_linear_left_alm.weight'] # Left unit to left output
        results['readout_right'] = weights_dict['readout_linear_right_alm.weight']  # Right unit to right output
    
    # Input projection weights
    results['input_left_to_left'] = weights_dict['w_xh_linear_left_alm.weight'] # Left input to left unit
    results['input_right_to_right'] = weights_dict['w_xh_linear_right_alm.weight']  # Right input to right unit
    
    # Recurrent weights
    results['recurrent_left_to_left'] = weights_dict['rnn_cell.w_hh_linear_ll.weight']
    results['recurrent_right_to_right'] = weights_dict['rnn_cell.w_hh_linear_rr.weight']
    results['recurrent_left_to_right'] = weights_dict['rnn_cell.w_hh_linear_lr.weight']
    results['recurrent_right_to_left'] = weights_dict['rnn_cell.w_hh_linear_rl.weight']
    
    # Key relationships in paths
    if configs['model_type'] == 'TwoHemiRNNTanh':
        results['left_path'] = results['input_left_to_left'] * results['readout_left']  # Left sensory → left unit → left output
        results['cross_path'] = results['input_right_to_left'] * results['readout_left']  # Right sensory → left unit → left output
    
    results['left_amp'] = left_amp
    results['right_amp'] = right_amp
    results['input_ratio'] = left_amp / right_amp if right_amp > 0 else float('inf')
    

    return results

def plot_results_single_readout(all_results):
    """Create comprehensive visualizations for single readout model"""
    # Extract data for plotting
    left_amps = [r['left_amp'] for r in all_results]
    right_amps = [r['right_amp'] for r in all_results]
    input_ratios = [r['input_ratio'] for r in all_results]
    
    readout_all = [r['output_ratio'][0] for r in all_results]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    # plt.title('Minimal 2-Unit Dual ALM RNN: Input Asymmetry Effects', fontsize=16)
    
    # Plot 1: Readout weights ratio vs input strength ratios
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(input_ratios, readout_all, 'ro-', label='Readout', linewidth=2, markersize=8)
    ax1.set_xlabel('Input Strength Ratio (Left/Right)')
    ax1.set_ylabel('Readout Weight')
    ax1.set_title('Readout Weights vs Input Strength Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('minimal_rnn_analysis_single_readout.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_results(all_results):
    """Create comprehensive visualizations"""
    # Extract data for plotting
    left_amps = [r['left_amp'] for r in all_results]
    right_amps = [r['right_amp'] for r in all_results]
    input_ratios = [r['input_ratio'] for r in all_results]
    
    readout_left = [r['readout_left'] for r in all_results]
    readout_right = [r['readout_right'] for r in all_results]
    
    left_path = [r['left_path'] for r in all_results]
    cross_path = [r['cross_path'] for r in all_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Minimal 2-Unit Dual ALM RNN: Input Asymmetry Effects', fontsize=16)
    
    # Plot 1: Readout weights vs input strength ratios
    ax1 = axes[0, 0]
    ax1.plot(input_ratios, readout_left, 'ro-', label='Left Hemisphere Readout', linewidth=2, markersize=8)
    ax1.plot(input_ratios, readout_right, 'bo-', label='Right Hemisphere Readout', linewidth=2, markersize=8)
    ax1.set_xlabel('Input Strength Ratio (Left/Right)')
    ax1.set_ylabel('Readout Weight')
    ax1.set_title('Readout Weights vs Input Strength Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Left vs Right readout weights
    ax2 = axes[0, 1]
    ax2.scatter(readout_left, readout_right, c=input_ratios, cmap='viridis', s=100, alpha=0.7)
    ax2.plot([min(readout_left), max(readout_left)], [min(readout_right), max(readout_right)], 'k--', alpha=0.5)
    ax2.set_xlabel('Left Hemisphere Readout Weight')
    ax2.set_ylabel('Right Hemisphere Readout Weight')
    ax2.set_title('Left vs Right Readout Weights')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sensory input pathways comparison
    ax3 = axes[1, 0]
    ax3.plot(input_ratios, left_path, 'go-', label='Left Sensory → Left Unit → Left Output', linewidth=2, markersize=8)
    ax3.plot(input_ratios, cross_path, 'mo-', label='Right Sensory → Left Unit → Left Output', linewidth=2, markersize=8)
    ax3.set_xlabel('Input Strength Ratio (Left/Right)')
    ax3.set_ylabel('Combined Weight')
    ax3.set_title('Sensory Input Pathway Weights')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Plot 4: Input projection weights heatmap
    ax4 = axes[1, 1]
    # Create a matrix of input weights for visualization
    input_weights_matrix = np.array([
        [all_results[4]['input_left_to_left'], all_results[4]['input_left_to_right']],  # Balanced case (1.0, 1.0)
        [all_results[4]['input_right_to_left'], all_results[4]['input_right_to_right']]
    ])
    
    im = ax4.imshow(input_weights_matrix, cmap='RdBu_r', aspect='auto')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Left Unit', 'Right Unit'])
    ax4.set_yticklabels(['Left Input', 'Right Input'])
    ax4.set_title('Input Projection Weights (Balanced Case)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Weight Value')
    
    plt.tight_layout()
    plt.savefig('minimal_rnn_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main analysis function"""
    print("Starting Minimal 2-Unit Dual ALM RNN Analysis")
    print("Using prebuilt functions from dual_alm_rnn_exp.py")
    
    # Initialize experiment
    exp = DualALMRNNExp()
    
    # Generate dataset if it doesn't exist
    if not os.path.exists(os.path.join(exp.configs['data_dir'], 'train', 'onehot_sensory_inputs.npy')):
        print("Generating one-hot dataset...")
        exp.generate_dataset_onehot()
    
    # Train RNN for each input asymmetry

    # Load results if they exist
    if os.path.exists('minimal_rnn_results.npy'):
        all_results = np.load('minimal_rnn_results.npy', allow_pickle=True).tolist()
        print("\nResults loaded from 'minimal_rnn_results.npy'")
    else:
        all_results = []
    
        for left_amp, right_amp in input_asym:
            print(f"\n{'='*60}")
            print(f"Testing: left_amp={left_amp:.1f}, right_amp={right_amp:.1f}")
            print(f"{'='*60}")
            
            # Train the RNN using prebuilt functions
            weights, train_losses, val_losses, train_scores, val_scores = train_minimal_rnn(
                exp.configs.copy(), left_amp, right_amp, exp
            )
            
            # Analyze weight relationships
            results = analyze_weight_relationships(exp.configs, weights, left_amp, right_amp)
            all_results.append(results)
            
            # Print key results
            # print(f"\nKey Results:")
            # print(f"  Left readout weight: {results['readout_left'].shape}")
            # print(f"  Right readout weight: {results['readout_right'].shape}")
            # print(f"  Left pathway weight: {results['left_path'].shape}")
            # print(f"  Cross pathway weight: {results['cross_path'].shape}")
        # Save results
        np.save('minimal_rnn_results.npy', all_results)
        print("\nResults saved to 'minimal_rnn_results.npy'")

    # Create visualizations
    print("\nCreating visualizations...")
    if exp.configs['model_type'] == 'TwoHemiRNNTanh_single_readout':
        fig = plot_results_single_readout(all_results)
    else:
        fig = plot_results(all_results)
    

    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for i, (left_amp, right_amp) in enumerate(input_asym):
        r = all_results[i]
        print(f"Left: {left_amp:.1f}, Right: {right_amp:.1f} | "
              f"Left Readout: {r['readout_left']:.4f}, Right Readout: {r['readout_right']:.4f} | "
              f"Ratio: {r['input_ratio']:.2f}")

if __name__ == "__main__":
    main()
