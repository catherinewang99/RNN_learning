#!/usr/bin/env python3
"""
Minimal 2-unit Dual ALM RNN Analysis
Analyzes how sensory input strength asymmetries affect readout weights and cross-hemisphere connections
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
    """Train a minimal RNN with specific input amplitudes"""
    print(f"\nTraining RNN with left_amp={left_amp:.1f}, right_amp={right_amp:.1f}")
    
    # Modify for minimal RNN
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
    device = torch.device("mps")  # Use CPU to avoid MPS issues
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
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['l2_weight_decay'])
    loss_fct = nn.BCEWithLogitsLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []
    
    for epoch in range(configs['n_epochs']):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            hs, zs = model(inputs)
            
            # Calculate loss
            loss = loss_fct(zs[:,-1,:], labels.float())
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            preds = (zs[:,-1,:] >= 0).long()
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            
            epoch_train_loss += loss.item()
            epoch_train_correct += correct
            epoch_train_total += total
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        epoch_val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                hs, zs = model(inputs)
                
                loss = loss_fct(zs[:,-1,:], labels.float())
                preds = (zs[:,-1,:] >= 0).long()
                correct = (preds == labels).sum().item()
                total = labels.size(0)
                
                epoch_val_loss += loss.item()
                epoch_val_correct += correct
                epoch_val_total += total
        
        # Record metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_acc = 100. * epoch_train_correct / epoch_train_total
        val_acc = 100. * epoch_val_correct / epoch_val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_scores.append(train_acc)
        val_scores.append(val_acc)
        
        print(f'Epoch {epoch+1}/{configs["n_epochs"]}: '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.1f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.1f}%')
    
    # Extract final weights
    final_weights = {}
    for name, param in model.named_parameters():
        final_weights[name] = param.data.cpu().numpy().copy()
    
    return final_weights, train_losses, val_losses, train_scores, val_scores

def analyze_weight_relationships(weights_dict, left_amp, right_amp):
    """Analyze the weight relationships of interest"""
    results = {}
    
    # Readout weights for each hemisphere
    results['readout_left'] = weights_dict['readout_linear_left_alm.weight'][0, 0]  # Left unit to left output
    results['readout_right'] = weights_dict['readout_linear_right_alm.weight'][0, 0]  # Right unit to right output
    
    # Input projection weights
    results['input_left_to_left'] = weights_dict['w_xh_linear_left_alm.weight'][0, 0]  # Left input to left unit
    results['input_left_to_right'] = weights_dict['w_xh_linear_left_alm.weight'][0, 1]  # Left input to right unit
    results['input_right_to_left'] = weights_dict['w_xh_linear_right_alm.weight'][0, 0]  # Right input to left unit
    results['input_right_to_right'] = weights_dict['w_xh_linear_right_alm.weight'][0, 1]  # Right input to right unit
    
    # Recurrent weights
    results['recurrent_left_to_left'] = weights_dict['rnn_cell.w_hh_linear_ll.weight'][0, 0]
    results['recurrent_right_to_right'] = weights_dict['rnn_cell.w_hh_linear_rr.weight'][0, 0]
    results['recurrent_left_to_right'] = weights_dict['rnn_cell.w_hh_linear_lr.weight'][0, 0]
    results['recurrent_right_to_left'] = weights_dict['rnn_cell.w_hh_linear_rl.weight'][0, 0]
    
    # Key relationships you mentioned
    results['left_path'] = results['input_left_to_left'] * results['readout_left']  # Left sensory → left unit → left output
    results['cross_path'] = results['input_right_to_left'] * results['readout_left']  # Right sensory → left unit → left output
    
    results['left_amp'] = left_amp
    results['right_amp'] = right_amp
    results['input_ratio'] = left_amp / right_amp if right_amp > 0 else float('inf')
    
    return results

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

    
    # Initialize experiment
    exp = DualALMRNNExp()
    
    # Generate dataset if it doesn't exist
    if not os.path.exists(os.path.join(exp.configs['data_dir'], 'train', 'onehot_sensory_inputs.npy')):
        print("Generating one-hot dataset...")
        exp.generate_dataset_onehot()
    
    # Train RNN for each input asymmetry
    all_results = []
    
    for left_amp, right_amp in input_asym:
        print(f"\n{'='*60}")
        print(f"Testing: left_amp={left_amp:.1f}, right_amp={right_amp:.1f}")
        print(f"{'='*60}")
        
        # Train the RNN
        weights, train_losses, val_losses, train_scores, val_scores = train_minimal_rnn(
            configs, left_amp, right_amp, exp
        )
        
        # Analyze weight relationships
        results = analyze_weight_relationships(weights, left_amp, right_amp)
        all_results.append(results)
        
        # Print key results
        print(f"\nKey Results:")
        print(f"  Left readout weight: {results['readout_left']:.4f}")
        print(f"  Right readout weight: {results['readout_right']:.4f}")
        print(f"  Left pathway weight: {results['left_path']:.4f}")
        print(f"  Cross pathway weight: {results['cross_path']:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = plot_results(all_results)
    
    # Save results
    np.save('minimal_rnn_results.npy', all_results)
    print("\nResults saved to 'minimal_rnn_results.npy'")
    
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
