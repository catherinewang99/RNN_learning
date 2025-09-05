import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
import os
import json
from dual_alm_rnn_models import TwoHemiRNNTanh_single_readout
from dual_alm_rnn_exp import DualALMRNNExp
import argparse, os, math, pickle, json

plt.rcParams['pdf.fonttype'] = '42' 

def load_model_weights(model_path, model_type):
    
    """
    Load weights from a trained model
    
    Args:
        model_path: Path to the model directory containing best_model.pth
    
    Returns:
        dict: Dictionary containing all weight matrices
    """
    # Load configs to get model parameters
    with open('dual_alm_rnn_configs.json', 'r') as f:
        configs = json.load(f)
    
    # Initialize model
    a = 0.5  # t_step/tau
    pert_begin = 56  # (pert_begin_t - trial_begin_t)//t_step
    pert_end = 88    # (pert_end_t - trial_begin_t)//t_step
    
    model = TwoHemiRNNTanh_single_readout(configs, a, pert_begin, pert_end)
    
    # Load trained weights
    # if os.path.exists(os.path.join(model_path, 'last_model.pth')):
    #     checkpoint_path = os.path.join(model_path, 'last_model.pth')
    # else:
    #     checkpoint_path = os.path.join(model_path, 'best_model.pth')
    checkpoint_path = os.path.join(model_path, model_type + '_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    
    # Extract weights
    weights = {}
    
    # Input weights (2x2 each for left and right)
    weights['input_left'] = model.w_xh_linear_left_alm.weight.data.cpu().numpy()  # (2, 2)
    weights['input_right'] = model.w_xh_linear_right_alm.weight.data.cpu().numpy()  # (2, 2)
    
    # Recurrent weights (2x2 each)
    weights['recurrent_ll'] = model.rnn_cell.w_hh_linear_ll.weight.data.cpu().numpy()  # (2, 2)
    weights['recurrent_rr'] = model.rnn_cell.w_hh_linear_rr.weight.data.cpu().numpy()  # (2, 2)
    weights['recurrent_lr'] = model.rnn_cell.w_hh_linear_lr.weight.data.cpu().numpy()  # (2, 2)
    weights['recurrent_rl'] = model.rnn_cell.w_hh_linear_rl.weight.data.cpu().numpy()  # (2, 2)
    
    # Readout weights (1x4)
    weights['readout'] = model.readout_linear.weight.data.cpu().numpy()  # (1, 4)
    
    return weights, configs

def visualize_rnn_weights(model_path, configs, model_type, save_path=None, figsize=(12, 12)):
    """
    Create a comprehensive visualization of RNN weights
    
    Args:
        model_path: Path to the model directory
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    # Load weights
    weights, configs = load_model_weights(model_path, model_type)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Define node positions
    # Sensory input layer (top) - closer together
    sensory_y = 8.5
    sensory_left_x = 3.5
    sensory_right_x = 6.5
    
    # Recurrent layer (middle) - single row with extra space between L and R
    recurrent_y = 5.0
    recurrent_x_positions = [1.5, 2.5, 7.5, 8.5]  # L1, L2, R1, R2
    
    # Readout layer (bottom)
    readout_y = 1.5
    readout_x = 5.0
    
    # Node properties
    node_radius = 0.3
    node_colors = {
        'sensory': '#E8E8E8',  # Light gray
        'recurrent_left': '#FFE6E6',  # Light red
        'recurrent_right': '#E6F3FF',  # Light blue
        'readout': '#F0F0F0'  # Light gray
    }
    
    # Draw sensory input nodes
    sensory_left = Circle((sensory_left_x, sensory_y), node_radius, 
                         facecolor=node_colors['sensory'], edgecolor='black', linewidth=2)
    sensory_right = Circle((sensory_right_x, sensory_y), node_radius,
                          facecolor=node_colors['sensory'], edgecolor='black', linewidth=2)
    ax.add_patch(sensory_left)
    ax.add_patch(sensory_right)
    
    # Add sensory labels
    ax.text(sensory_left_x, sensory_y, 'L', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(sensory_right_x, sensory_y, 'R', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Draw recurrent nodes
    recurrent_nodes = []
    for i, x in enumerate(recurrent_x_positions):
        if i < 2:  # Left hemisphere
            color = node_colors['recurrent_left']
            label = f'L{i+1}'
        else:  # Right hemisphere
            color = node_colors['recurrent_right']
            label = f'R{i-1}'
        
        node = Circle((x, recurrent_y), node_radius,
                     facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(node)
        recurrent_nodes.append(node)
        
        # Add labels
        ax.text(x, recurrent_y, label, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw readout node
    readout_node = Circle((readout_x, readout_y), node_radius,
                         facecolor=node_colors['readout'], edgecolor='black', linewidth=2)
    ax.add_patch(readout_node)
    ax.text(readout_x, readout_y, 'OUT', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Helper function to get arrow properties
    def get_arrow_props(weight_value, max_weight=None):
        if max_weight is None:
            max_weight = max(np.abs(weights['input_left'].max()), 
                           np.abs(weights['input_right'].max()),
                           np.abs(weights['recurrent_ll'].max()),
                           np.abs(weights['recurrent_rr'].max()),
                           np.abs(weights['recurrent_lr'].max()),
                           np.abs(weights['recurrent_rl'].max()),
                           np.abs(weights['readout'].max()))
        
        # Linear scale for thickness with wider range
        abs_weight = abs(weight_value)
        if abs_weight < 1e-6:
            thickness = 0.1
        else:
            # Linear scale: thickness ranges from 0.1 to 8.0
            normalized = abs_weight / max_weight
            thickness = 0.1 + normalized * 7.9
            thickness = max(0.1, min(8.0, thickness))  # Clamp between 0.1 and 8.0
        
        # Color based on sign
        color = 'blue' if weight_value < 0 else 'red'
        
        return thickness, color
    
    # Get max weight for normalization
    all_weights = np.concatenate([
        weights['input_left'].flatten(),
        weights['input_right'].flatten(),
        weights['recurrent_ll'].flatten(),
        weights['recurrent_rr'].flatten(),
        weights['recurrent_lr'].flatten(),
        weights['recurrent_rl'].flatten(),
        weights['readout'].flatten()
    ])
    # Calculate weight range for scaling (excluding intentionally zero cross-hemisphere weights)
    # Exclude w_hh_linear_lr and w_hh_linear_rl which are set to 0 by design
    non_zero_weights = np.concatenate([
        weights['input_left'].flatten(),
        weights['input_right'].flatten(),
        weights['recurrent_ll'].flatten(),
        weights['recurrent_rr'].flatten(),
        # Exclude recurrent_lr and recurrent_rl (cross-hemisphere weights set to 0)
        weights['readout'].flatten()
    ])
    
    max_weight = np.max(np.abs(non_zero_weights))
    min_weight = np.min(np.abs(non_zero_weights))
    
    # Draw input connections (8 arrows total: L/R sensory inputs to all 4 neurons)
    # w_xh_linear_left_alm: (2, 2) - 2D input to 2 left hemisphere neurons
    # w_xh_linear_right_alm: (2, 2) - 2D input to 2 right hemisphere neurons
    # Both weight matrices take the same 2D one-hot input [left_channel, right_channel]
    input_connections = [
        # Left sensory input (channel 0) to all 4 neurons
        (sensory_left_x, sensory_y, recurrent_x_positions[0], recurrent_y, weights['input_left'][0, 0]),  # L->L1
        (sensory_left_x, sensory_y, recurrent_x_positions[1], recurrent_y, weights['input_left'][0, 1]),  # L->L2
        (sensory_left_x, sensory_y, recurrent_x_positions[2], recurrent_y, weights['input_right'][0, 0]),  # L->R1
        (sensory_left_x, sensory_y, recurrent_x_positions[3], recurrent_y, weights['input_right'][0, 1]),  # L->R2
        
        # Right sensory input (channel 1) to all 4 neurons
        (sensory_right_x, sensory_y, recurrent_x_positions[0], recurrent_y, weights['input_left'][1, 0]),  # R->L1
        (sensory_right_x, sensory_y, recurrent_x_positions[1], recurrent_y, weights['input_left'][1, 1]),  # R->L2
        (sensory_right_x, sensory_y, recurrent_x_positions[2], recurrent_y, weights['input_right'][1, 0]),  # R->R1
        (sensory_right_x, sensory_y, recurrent_x_positions[3], recurrent_y, weights['input_right'][1, 1]),  # R->R2
    ]
    
    for x1, y1, x2, y2, weight in input_connections:
        thickness, color = get_arrow_props(weight, max_weight)
        if thickness > 0.1:  # Only draw if weight is significant
            arrow = FancyArrowPatch((x1, y1-node_radius), (x2, y2+node_radius),
                                  arrowstyle='->', mutation_scale=20, 
                                  linewidth=thickness, color=color, alpha=0.8)
            ax.add_patch(arrow)
    
    # Draw recurrent connections (16 arrows total: 4x4 matrix for each recurrent weight type)
    recurrent_connections = [
        # Left-to-left connections
        (recurrent_x_positions[0], recurrent_y, recurrent_x_positions[0], recurrent_y, weights['recurrent_ll'][0, 0]),  # L1->L1
        (recurrent_x_positions[0], recurrent_y, recurrent_x_positions[1], recurrent_y, weights['recurrent_ll'][0, 1]),  # L1->L2
        (recurrent_x_positions[1], recurrent_y, recurrent_x_positions[0], recurrent_y, weights['recurrent_ll'][1, 0]),  # L2->L1
        (recurrent_x_positions[1], recurrent_y, recurrent_x_positions[1], recurrent_y, weights['recurrent_ll'][1, 1]),  # L2->L2
        
        # Right-to-right connections
        (recurrent_x_positions[2], recurrent_y, recurrent_x_positions[2], recurrent_y, weights['recurrent_rr'][0, 0]),  # R1->R1
        (recurrent_x_positions[2], recurrent_y, recurrent_x_positions[3], recurrent_y, weights['recurrent_rr'][0, 1]),  # R1->R2
        (recurrent_x_positions[3], recurrent_y, recurrent_x_positions[2], recurrent_y, weights['recurrent_rr'][1, 0]),  # R2->R1
        (recurrent_x_positions[3], recurrent_y, recurrent_x_positions[3], recurrent_y, weights['recurrent_rr'][1, 1]),  # R2->R2
        
        # Left-to-right connections
        (recurrent_x_positions[0], recurrent_y, recurrent_x_positions[2], recurrent_y, weights['recurrent_lr'][0, 0]),  # L1->R1
        (recurrent_x_positions[0], recurrent_y, recurrent_x_positions[3], recurrent_y, weights['recurrent_lr'][0, 1]),  # L1->R2
        (recurrent_x_positions[1], recurrent_y, recurrent_x_positions[2], recurrent_y, weights['recurrent_lr'][1, 0]),  # L2->R1
        (recurrent_x_positions[1], recurrent_y, recurrent_x_positions[3], recurrent_y, weights['recurrent_lr'][1, 1]),  # L2->R2
        
        # Right-to-left connections
        (recurrent_x_positions[2], recurrent_y, recurrent_x_positions[0], recurrent_y, weights['recurrent_rl'][0, 0]),  # R1->L1
        (recurrent_x_positions[2], recurrent_y, recurrent_x_positions[1], recurrent_y, weights['recurrent_rl'][0, 1]),  # R1->L2
        (recurrent_x_positions[3], recurrent_y, recurrent_x_positions[0], recurrent_y, weights['recurrent_rl'][1, 0]),  # R2->L1
        (recurrent_x_positions[3], recurrent_y, recurrent_x_positions[1], recurrent_y, weights['recurrent_rl'][1, 1]),  # R2->L2
    ]
    
    for x1, y1, x2, y2, weight in recurrent_connections:
        thickness, color = get_arrow_props(weight, max_weight)
        if thickness > 0.1:  # Only draw if weight is significant
            # For self-connections, draw a small loop outside the node
            if x1 == x2 and y1 == y2:
                # Self-connection: start and end on the circle, but midpoint extends outward
                # Start at 0 degrees (right side), end at 90 degrees (top) - both on the circle
                start_x = x1 + node_radius * np.cos(0)  # 0 degrees (right side) - ON the circle
                start_y = y1 + node_radius * np.sin(0)
                end_x = x1 + node_radius * np.cos(np.pi/2)  # 90 degrees (top) - ON the circle
                end_y = y1 + node_radius * np.sin(np.pi/2)
                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                      arrowstyle='->', mutation_scale=15,
                                      linewidth=thickness, color=color, alpha=0.8,
                                      connectionstyle="arc3,rad=0.8")
            else:
                # Regular connection
                arrow = FancyArrowPatch((x1, y1-node_radius), (x2, y2+node_radius),
                                      arrowstyle='->', mutation_scale=20,
                                      linewidth=thickness, color=color, alpha=0.8)
            ax.add_patch(arrow)
    
    # Draw readout connections (4 arrows: each recurrent unit to readout)
    readout_connections = [
        (recurrent_x_positions[0], recurrent_y, readout_x, readout_y, weights['readout'][0, 0]),  # L1->OUT
        (recurrent_x_positions[1], recurrent_y, readout_x, readout_y, weights['readout'][0, 1]),  # L2->OUT
        (recurrent_x_positions[2], recurrent_y, readout_x, readout_y, weights['readout'][0, 2]),  # R1->OUT
        (recurrent_x_positions[3], recurrent_y, readout_x, readout_y, weights['readout'][0, 3]),  # R2->OUT
    ]
    
    for x1, y1, x2, y2, weight in readout_connections:
        thickness, color = get_arrow_props(weight, max_weight)
        if thickness > 0.1:  # Only draw if weight is significant
            arrow = FancyArrowPatch((x1, y1-node_radius), (x2, y2+node_radius),
                                  arrowstyle='->', mutation_scale=20,
                                  linewidth=thickness, color=color, alpha=0.8)
            ax.add_patch(arrow)
    

    # Add title
    ax.text(5, 9.5, 'RNN Weight Visualization: L={}, R={}'.format(configs['xs_left_alm_amp'], configs['xs_right_alm_amp']), ha='center', va='center', 
            fontsize=16, fontweight='bold')
    
    # Add reference arrows showing thickest and thinnest weights
    fig = plt.gcf()
    
    # Position for reference arrows (bottom left corner)
    ref_x_start = 0.5
    ref_y = 0.5
    
    # Draw thickest weight reference arrow
    thickest_thickness, _ = get_arrow_props(max_weight, max_weight)
    thickest_arrow = FancyArrowPatch((ref_x_start, ref_y), (ref_x_start + 0.8, ref_y),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=thickest_thickness, color='black', alpha=0.8)
    ax.add_patch(thickest_arrow)
    
    # Add text for thickest weight
    ax.text(ref_x_start + 0.9, ref_y, f'{max_weight:.3f}', fontsize=9, 
            verticalalignment='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Draw thinnest weight reference arrow (slightly below)
    thinnest_thickness, _ = get_arrow_props(min_weight, max_weight)
    thinnest_arrow = FancyArrowPatch((ref_x_start, ref_y - 0.3), (ref_x_start + 0.8, ref_y - 0.3),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=thinnest_thickness, color='black', alpha=0.8)
    ax.add_patch(thinnest_arrow)
    
    # Add text for thinnest weight
    ax.text(ref_x_start + 0.9, ref_y - 0.3, f'{min_weight:.3f}', fontsize=9, 
            verticalalignment='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add color legend for positive/negative
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=3, label='Positive weights'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Negative weights')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add thickness explanation
    thickness_text = 'Arrow thickness ∝ |weight| (linear scale: 0.1 to 8.0)'
    ax.text(0.02, 0.08, thickness_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return weights

def find_model_paths():
    """
    Find all available model paths in the models directory
    """
    models_dir = 'dual_alm_rnn_models/TwoHemiRNNTanh_single_readout'
    if not os.path.exists(models_dir):
        return []
    
    model_paths = []
    for root, dirs, files in os.walk(models_dir):
        if 'best_model.pth' in files:
            model_paths.append(root)
    
    return model_paths
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--best', action='store_true', default=False)
    parser.add_argument('--last', action='store_true', default=False)
    return parser.parse_args()

def main():
    """
    Main function to run the visualization
    """
    # Find available model paths
    model_paths = find_model_paths()
    
    if not model_paths:
        print("No trained models found in dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/")
        print("Please train a model first or update the model_path variable")
        return
    
    # Use the first available model (you can modify this to choose a specific one)
    model_path = model_paths[0]

    # model_path='dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/train_type_modular/onehot/n_neurons_4_random_seed_0/n_epochs_30_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.20/init_cross_hemi_rel_factor_0.20'
    # model_path='dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/train_type_modular_corruption/onehot_cor_type_gaussian_epoch_10_noise_1.40/n_neurons_4_random_seed_0/n_epochs_30_n_epochs_across_hemi_0/lr_3.0e-03_bs_75/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_0.20/init_cross_hemi_rel_factor_0.20'
    with open('dual_alm_rnn_configs.json', 'r') as f:
        configs = json.load(f)
    
    exp = DualALMRNNExp()
    exp.configs = configs
    exp.init_sub_path(configs['train_type'])
    model_path = os.path.join(configs['models_dir'], configs['model_type'], exp.sub_path)

    print(f"Using model: {model_path}")
    
    # Create visualization
    save_path = 'figs/rnn_L{}_R{}_weights_visualization_{}.pdf'.format(exp.configs['xs_left_alm_amp'], exp.configs['xs_right_alm_amp'], exp.configs['train_type'])

    args = parse_args()


    if args.best:
        weights = visualize_rnn_weights(model_path, configs, 'best', save_path=save_path)
    elif args.last:
        weights = visualize_rnn_weights(model_path, configs, 'last', save_path=save_path)
    
    # Print weight summary
    print("\nWeight Summary:")
    print("="*50)
    for name, weight_matrix in weights.items():
        print(f"{name}: shape {weight_matrix.shape}, range [{weight_matrix.min():.3f}, {weight_matrix.max():.3f}]")
    
    # Print available model paths for reference
    # if len(model_paths) > 1:
    #     print(f"\nOther available models ({len(model_paths)-1} more):")
    #     for i, path in enumerate(model_paths[1:], 1):
    #         print(f"  {i}. {path}")

if __name__ == "__main__":
    main()
