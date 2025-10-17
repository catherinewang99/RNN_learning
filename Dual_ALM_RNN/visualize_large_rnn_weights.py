import torch, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
import os
import json
from dual_alm_rnn_models import *
from dual_alm_rnn_exp import DualALMRNNExp
import argparse, os, math, pickle, json

plt.rcParams['pdf.fonttype'] = '42' 

def load_model_weights(model_path, mode):
    
    """
    Load weights from a trained model and extract weights for first/last units from each hemisphere
    
    Args:
        model_path: Path to the model directory containing best_model.pth
        mode: 'best', 'last', or 'epoch_<N>'
    
    Returns:
        dict: Dictionary containing weight matrices for selected units
    """
    # Load configs to get model parameters
    with open('dual_alm_rnn_configs.json', 'r') as f:
        configs = json.load(f)
    
    # Update configs to match the model before initializing
    # configs['n_neurons'] = 256
    # configs['xs_left_alm_amp'] = 1.0
    # configs['xs_right_alm_amp'] = 1.0
    # configs['random_seed'] = 4
    # Don't override train_type - use the original config value
    
    # Initialize model
    a = 0.5  # t_step/tau
    pert_begin = 56  # (pert_begin_t - trial_begin_t)//t_step
    pert_end = 88    # (pert_end_t - trial_begin_t)//t_step
    
    model = TwoHemiRNNTanh_single_readout(configs, a, pert_begin, pert_end)
    
    # Resolve checkpoint path from mode: 'best', 'last', or 'epoch_<N>'
    if mode == 'best':
        checkpoint_file = 'best_model.pth'
    elif mode == 'last':
        checkpoint_file = 'last_model.pth'
    elif isinstance(mode, str) and mode.startswith('epoch_'):
        try:
            epoch_num = int(mode.split('_', 1)[1])
            checkpoint_file = f'model_epoch_{epoch_num}.pth'
        except Exception:
            checkpoint_file = 'last_model.pth'
    else:
        checkpoint_file = 'last_model.pth'
    checkpoint_path = os.path.join(model_path, checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Get actual model dimensions from the state dict
    # The readout weight shape tells us the total number of neurons
    readout_weight_shape = state_dict['readout_linear.weight'].shape
    n_neurons = readout_weight_shape[1]  # (1, n_neurons)
    n_left = n_neurons // 2
    n_right = n_neurons - n_left
    
    # Check input weight dimensions
    input_left_shape = state_dict['w_xh_linear_left_alm.weight'].shape
    input_right_shape = state_dict['w_xh_linear_right_alm.weight'].shape
    print(f"Actual input weight shapes: left {input_left_shape}, right {input_right_shape}")
    print(f"Expected n_left: {n_left}, n_right: {n_right}")
    
    # Update configs with actual model dimensions
    configs['n_neurons'] = n_neurons
    
    model.load_state_dict(state_dict)
    
    print(f"Model has {n_neurons} total neurons: {n_left} left, {n_right} right")
    
    # Define selected unit indices (first 2 and last 2 from each hemisphere)
    if n_left >= 4:
        left_selected = [0, 1, n_left-2, n_left-1]  # First 2 and last 2
    else:
        left_selected = list(range(n_left))  # All units if < 4
        
    if n_right >= 4:
        right_selected = [0, 1, n_right-2, n_right-1]  # First 2 and last 2
    else:
        right_selected = list(range(n_right))  # All units if < 4
    
    print(f"Selected left units: {left_selected}")
    print(f"Selected right units: {right_selected}")
    print(f"Total selected units: {len(left_selected) + len(right_selected)}")
    
    # Extract weights
    weights = {}
    
    # Input weights - extract for selected units
    input_left_full = model.w_xh_linear_left_alm.weight.data.cpu().numpy()  # (n_left, 2) for onehot
    input_right_full = model.w_xh_linear_right_alm.weight.data.cpu().numpy()  # (n_right, 2) for onehot
    
    weights['input_left'] = input_left_full[left_selected, :]  # (4, 2)
    weights['input_right'] = input_right_full[right_selected, :]  # (4, 2)
    
    # Recurrent weights - extract submatrices for selected units
    recurrent_ll_full = model.rnn_cell.w_hh_linear_ll.weight.data.cpu().numpy()  # (n_left, n_left)
    recurrent_rr_full = model.rnn_cell.w_hh_linear_rr.weight.data.cpu().numpy()  # (n_right, n_right)
    recurrent_lr_full = model.rnn_cell.w_hh_linear_lr.weight.data.cpu().numpy()  # (n_left, n_right)
    recurrent_rl_full = model.rnn_cell.w_hh_linear_rl.weight.data.cpu().numpy()  # (n_right, n_left)
    
    # Extract submatrices for selected units
    weights['recurrent_ll'] = recurrent_ll_full[np.ix_(left_selected, left_selected)]  # (4, 4)
    weights['recurrent_rr'] = recurrent_rr_full[np.ix_(right_selected, right_selected)]  # (4, 4)
    weights['recurrent_lr'] = recurrent_lr_full[np.ix_(left_selected, right_selected)]  # (4, 4)
    weights['recurrent_rl'] = recurrent_rl_full[np.ix_(right_selected, left_selected)]  # (4, 4)
    
    # Readout weights - extract for selected units (TwoHemiRNNTanh_single_readout has single readout)
    readout_full = model.readout_linear.weight.data.cpu().numpy()  # (1, n_neurons)
    selected_readout_indices = left_selected + [n_left + i for i in right_selected]
    weights['readout'] = readout_full[:, selected_readout_indices]  # (1, 8)
    
    # Extract bias values for coloring nodes
    biases = {}
    biases['readout'] = model.readout_linear.bias.data.cpu().numpy()  # (1,)
    
    # Extract RNN cell biases for selected units
    biases_left_full = model.rnn_cell.w_hh_linear_ll.bias.data.cpu().numpy()  # (n_left,)
    biases_right_full = model.rnn_cell.w_hh_linear_rr.bias.data.cpu().numpy()  # (n_right,)
    
    biases['left_hemi'] = biases_left_full[left_selected]  # (4,)
    biases['right_hemi'] = biases_right_full[right_selected]  # (4,)
    
    print(f"Readout bias: {biases['readout']}")
    print(f"Selected left hemisphere biases: {biases['left_hemi']}")
    print(f"Selected right hemisphere biases: {biases['right_hemi']}")
    return weights, configs, biases

def visualize_rnn_weights(model_path, configs, mode, save_path=None, figsize=(16, 12), show=True):
    """
    Create a comprehensive visualization of RNN weights for large models
    Shows first 2 and last 2 units from each hemisphere (8 total units)
    
    Args:
        model_path: Path to the model directory
        configs: Model configuration dictionary
        mode: 'best', 'last', or 'epoch_<N>'
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        show: Whether to display the plot
    """
    # Load weights and biases
    weights, configs, biases = load_model_weights(model_path, mode)
    
    # Get the selected units from the loaded weights
    n_neurons = configs['n_neurons']
    n_left = n_neurons // 2
    n_right = n_neurons - n_left
    
    # Define selected unit indices (first 2 and last 2 from each hemisphere)
    if n_left >= 4:
        left_selected = [0, 1, n_left-2, n_left-1]  # First 2 and last 2
    else:
        left_selected = list(range(n_left))  # All units if < 4
        
    if n_right >= 4:
        right_selected = [0, 1, n_right-2, n_right-1]  # First 2 and last 2
    else:
        right_selected = list(range(n_right))  # All units if < 4
    
    # Create figure with wider aspect ratio for 8 units
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 14)  # Wider to accommodate 8 units
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Define node positions
    # Sensory input layer (top) - closer together
    sensory_y = 7.5
    sensory_left_x = 5.0
    sensory_right_x = 9.0
    
    # Recurrent layer (middle) - dynamic positioning based on number of units
    recurrent_y = 5.0
    n_left_selected = len(left_selected)
    n_right_selected = len(right_selected)
    n_total_selected = n_left_selected + n_right_selected
    
    # Calculate positions dynamically
    if n_total_selected <= 4:
        # For small models (4 units or less), use compact layout
        recurrent_x_positions = [2.0, 4.0, 8.0, 10.0][:n_total_selected]
    else:
        # For larger models (8 units), use the original layout
        recurrent_x_positions = [1.0, 2.5, 4.0, 5.5, 8.5, 10.0, 11.5, 13.0]
    
    # Readout layer (bottom) - symmetric distance from recurrent layer
    readout_y = 2.5
    readout_x = 7.0
    
    # Node properties
    node_radius = 0.5
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
    ax.text(sensory_left_x, sensory_y, 'C0', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(sensory_right_x, sensory_y, 'C1', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Calculate bias magnitude range for alpha scaling
    all_bias_values = np.concatenate([
        biases['left_hemi'],
        biases['right_hemi'], 
        biases['readout']
    ])
    max_bias_magnitude = np.max(np.abs(all_bias_values))
    min_bias_magnitude = np.min(np.abs(all_bias_values))
    
    # Helper function to calculate alpha based on bias magnitude
    def get_bias_alpha(bias_value, min_mag, max_mag):
        if max_mag == min_mag:  # All biases have same magnitude
            return 0.7
        # Scale alpha from 0.3 to 1.0 based on bias magnitude
        magnitude = np.abs(bias_value)
        alpha = 0.3 + 0.7 * (magnitude - min_mag) / (max_mag - min_mag)
        return np.clip(alpha, 0.3, 1.0)
    
    # Draw recurrent nodes with bias-based coloring and alpha
    recurrent_nodes = []
    for i, x in enumerate(recurrent_x_positions):
        if i < n_left_selected:  # Left hemisphere units
            if i < 2:
                label = f'L{i+1}'  # L1, L2
            else:
                label = f'L{i+125}'  # L127, L128 (approximate for visualization)
            # Use individual RNN cell bias for coloring (red for positive, blue for negative)
            bias_value = biases['left_hemi'][i]  # Individual bias for each left neuron
            color = 'red' if bias_value >= 0 else 'blue'
            alpha = get_bias_alpha(bias_value, min_bias_magnitude, max_bias_magnitude)
        else:  # Right hemisphere units
            right_idx = i - n_left_selected
            if right_idx < 2:
                label = f'R{right_idx+1}'  # R1, R2
            else:
                label = f'R{right_idx+125}'  # R127, R128 (approximate for visualization)
            # Use individual RNN cell bias for coloring (red for positive, blue for negative)
            bias_value = biases['right_hemi'][right_idx]  # Individual bias for each right neuron
            color = 'red' if bias_value >= 0 else 'blue'
            alpha = get_bias_alpha(bias_value, min_bias_magnitude, max_bias_magnitude)
        
        node = Circle((x, recurrent_y), node_radius,
                     facecolor=color, edgecolor='black', linewidth=2, alpha=alpha)
        ax.add_patch(node)
        recurrent_nodes.append(node)
        
        # Add labels
        ax.text(x, recurrent_y, label, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Draw readout node with bias-based coloring and alpha
    readout_bias = biases['readout'][0]  # Single bias value for readout
    readout_color = 'red' if readout_bias >= 0 else 'blue'
    readout_alpha = get_bias_alpha(readout_bias, min_bias_magnitude, max_bias_magnitude)
    readout_node = Circle((readout_x, readout_y), node_radius,
                         facecolor=readout_color, edgecolor='black', linewidth=2, alpha=readout_alpha)
    ax.add_patch(readout_node)
    ax.text(readout_x, readout_y, 'OUT', ha='center', va='center', fontsize=16, fontweight='bold')
    
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
            thickness = 0.3 + normalized * 11.7
            thickness = max(0.3, min(12.0, thickness))  # Clamp between 0.3 and 12.0
        
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
    # Scale input weights by amplitude coefficients for visualization
    left_amp = configs['xs_left_alm_amp']
    right_amp = configs['xs_right_alm_amp']
    
    # Create scaled input weights for visualization
    scaled_input_left = weights['input_left'] * left_amp
    scaled_input_right = weights['input_right'] * right_amp
    
    # Check if model type includes "cross_hemi" to determine whether to draw cross-hemispheric connections
    model_type = configs.get('model_type', '')
    train_type = configs.get('train_type', '')
    has_cross_hemi = 'cross_hemi' in train_type
    
    print(f"Model type: {model_type}")
    print(f"Train type: {train_type}")
    print(f"Has cross-hemispheric connections: {has_cross_hemi}")
    
    # Calculate weight range for scaling
    # Include all weights for proper thickness calculation
    # Cross-hemisphere weights may be non-zero depending on init_cross_hemi_rel_factor
    non_zero_weights = np.concatenate([
        scaled_input_left.flatten(),
        scaled_input_right.flatten(),
        weights['recurrent_ll'].flatten(),
        weights['recurrent_rr'].flatten(),
        weights['readout'].flatten()
    ])
    
    # Only include cross-hemispheric weights in range calculation if they should be drawn
    if has_cross_hemi:
        non_zero_weights = np.concatenate([
            non_zero_weights,
            weights['recurrent_lr'].flatten(),  # Include cross-hemisphere weights
            weights['recurrent_rl'].flatten(),  # Include cross-hemisphere weights
        ])
    
    max_weight = np.max(np.abs(non_zero_weights))
    min_weight = np.min(np.abs(non_zero_weights))
    
    # Draw input connections (dynamic based on number of units)
    # w_xh_linear_left_alm: (n_left_selected, 2) - selected left hemisphere neurons, 2 input channels
    # w_xh_linear_right_alm: (n_right_selected, 2) - selected right hemisphere neurons, 2 input channels
    # Both weight matrices take 2D one-hot input [left_channel, right_channel]
    # Use pre-calculated scaled weights for proper thickness visualization
    input_connections = []
    
    # Left sensory input (channel 0) to all units
    for i in range(n_left_selected):
        input_connections.append((
            sensory_left_x, sensory_y, recurrent_x_positions[i], recurrent_y, 
            scaled_input_left[i, 0]
        ))
    for i in range(n_right_selected):
        input_connections.append((
            sensory_left_x, sensory_y, recurrent_x_positions[i + n_left_selected], recurrent_y, 
            scaled_input_right[i, 0]
        ))
    
    # Right sensory input (channel 1) to all units
    for i in range(n_left_selected):
        input_connections.append((
            sensory_right_x, sensory_y, recurrent_x_positions[i], recurrent_y, 
            scaled_input_left[i, 1]
        ))
    for i in range(n_right_selected):
        input_connections.append((
            sensory_right_x, sensory_y, recurrent_x_positions[i + n_left_selected], recurrent_y, 
            scaled_input_right[i, 1]
        ))
    
    for x1, y1, x2, y2, weight in input_connections:
        thickness, color = get_arrow_props(weight, max_weight)
        if thickness > 0.1:  # Only draw if weight is significant
            arrow = FancyArrowPatch((x1, y1-node_radius), (x2, y2+node_radius),
                                  arrowstyle='->', mutation_scale=20, 
                                  linewidth=thickness, color=color, alpha=0.8)
            ax.add_patch(arrow)
    
    # Draw recurrent connections (dynamic based on number of units)
    recurrent_connections = []
    
    # Left-to-left connections
    for i in range(n_left_selected):
        for j in range(n_left_selected):
            recurrent_connections.append((
                recurrent_x_positions[i], recurrent_y, 
                recurrent_x_positions[j], recurrent_y, 
                weights['recurrent_ll'][i, j]
            ))
    
    # Right-to-right connections
    for i in range(n_right_selected):
        for j in range(n_right_selected):
            recurrent_connections.append((
                recurrent_x_positions[i + n_left_selected], recurrent_y, 
                recurrent_x_positions[j + n_left_selected], recurrent_y, 
                weights['recurrent_rr'][i, j]
            ))
    
    # Only draw cross-hemispheric connections if the model type includes "cross_hemi"
    if has_cross_hemi:
        # Left-to-right connections
        for i in range(n_left_selected):
            for j in range(n_right_selected):
                recurrent_connections.append((
                    recurrent_x_positions[i], recurrent_y, 
                    recurrent_x_positions[j + n_left_selected], recurrent_y, 
                    weights['recurrent_lr'][i, j]
                ))
        
        # Right-to-left connections
        for i in range(n_right_selected):
            for j in range(n_left_selected):
                recurrent_connections.append((
                    recurrent_x_positions[i + n_left_selected], recurrent_y, 
                    recurrent_x_positions[j], recurrent_y, 
                    weights['recurrent_rl'][i, j]
                ))
    
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
    
    # Draw readout connections (dynamic based on number of units)
    readout_connections = []
    for i in range(n_total_selected):
        readout_connections.append((
            recurrent_x_positions[i], recurrent_y, 
            readout_x, readout_y, 
            weights['readout'][0, i]
        ))
    
    for x1, y1, x2, y2, weight in readout_connections:
        thickness, color = get_arrow_props(weight, max_weight)
        if thickness > 0.1:  # Only draw if weight is significant
            arrow = FancyArrowPatch((x1, y1-node_radius), (x2, y2+node_radius),
                                  arrowstyle='->', mutation_scale=20,
                                  linewidth=thickness, color=color, alpha=0.8)
            ax.add_patch(arrow)
    

    # Add title
    cross_hemi_text = " (with cross-hemi)" if has_cross_hemi else " (no cross-hemi)"
    ax.text(7, 9.5, f'Large RNN Weight Visualization{cross_hemi_text}: L={configs["xs_left_alm_amp"]}, R={configs["xs_right_alm_amp"]}', ha='center', va='center', 
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
    
    # Add color legend for positive/negative weights and biases
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=3, label='Positive weights/bias'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Negative weights/bias')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add thickness explanation
    cross_hemi_note = "\nCross-hemispheric connections: shown" if has_cross_hemi else "\nCross-hemispheric connections: not shown"
    thickness_text = f'Arrow thickness ∝ |weight| (linear scale: 0.3 to 12.0)\nInput weights scaled by amp: L={left_amp:.1f}, R={right_amp:.1f}\nNode color: red=positive bias, blue=negative bias\nNode alpha: darker = larger |bias|\nShows first 2 and last 2 units from each hemisphere{cross_hemi_note}'
    ax.text(0.02, 0.08, thickness_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Print bias values and alpha for each recurrent node
    print("\nBias values and alpha for selected recurrent nodes:")
    for i in range(n_left_selected):
        print(f"L{i+1} bias: {biases['left_hemi'][i]:.6f}, alpha: {get_bias_alpha(biases['left_hemi'][i], min_bias_magnitude, max_bias_magnitude):.3f}")
    for i in range(n_right_selected):
        print(f"R{i+1} bias: {biases['right_hemi'][i]:.6f}, alpha: {get_bias_alpha(biases['right_hemi'][i], min_bias_magnitude, max_bias_magnitude):.3f}")
    print(f"Readout bias: {biases['readout'][0]:.6f}, alpha: {get_bias_alpha(biases['readout'][0], min_bias_magnitude, max_bias_magnitude):.3f}")
    print(f"Bias magnitude range: {min_bias_magnitude:.6f} to {max_bias_magnitude:.6f}")
    
    # Print input weight scaling information
    print(f"\nInput weight scaling:")
    print(f"Left amplitude coefficient: {left_amp:.3f}")
    print(f"Right amplitude coefficient: {right_amp:.3f}")
    print(f"Original input weights (left): {weights['input_left']}")
    print(f"Scaled input weights (left): {scaled_input_left}")
    print(f"Original input weights (right): {weights['input_right']}")
    print(f"Scaled input weights (right): {scaled_input_right}")
    print(f"Weight range (scaled): {min_weight:.6f} to {max_weight:.6f}")
    
    # Print cross-hemisphere weight information (only if cross-hemi connections are drawn)
    if has_cross_hemi:
        print(f"\nCross-hemisphere weights:")
        print(f"Left-to-right weights (LR): {weights['recurrent_lr']}")
        print(f"Right-to-left weights (RL): {weights['recurrent_rl']}")
        print(f"LR weight range: {np.min(np.abs(weights['recurrent_lr'])):.6f} to {np.max(np.abs(weights['recurrent_lr'])):.6f}")
        print(f"RL weight range: {np.min(np.abs(weights['recurrent_rl'])):.6f} to {np.max(np.abs(weights['recurrent_rl'])):.6f}")
    else:
        print(f"\nCross-hemisphere weights: Not drawn (model type does not include 'cross_hemi')")
    
    # Only apply layout/show if requested
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    elif show:
        plt.tight_layout()
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
    parser.add_argument('--epoch', type=int, default=None, help='Use model_epoch_<N>.pth')
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
    
    # Use a specific 256-neuron model for testing (with cross_hemi)
    # model_path = 'dual_alm_rnn_models/TwoHemiRNNTanh_single_readout/train_type_modular_fixed_input/n_neurons_256_random_seed_4/n_epochs_30_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_1.00_right_alm_amp_1.00/init_cross_hemi_rel_factor_0.20'
    exp = DualALMRNNExp()

    with open('dual_alm_rnn_configs.json', 'r') as f:
        configs = json.load(f)
    
    # Don't override train_type - use the original config value
    
    # if 'cross_hemi' in configs['train_type']:
        # model_path = 'dual_alm_rnn_models/{}/{}/n_neurons_256_random_seed_{}/unfix_epoch_{}/n_epochs_{}_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/'.format(
        #         configs['model_type'],
        #         configs['train_type'],
        #         configs['random_seed'],
        #         configs['unfix_epoch'],
        #         float(configs['xs_left_alm_amp']),
        #         float(configs['xs_right_alm_amp'])
        #     )

    model = getattr(sys.modules[__name__], configs['model_type'])(configs, \
        exp.a, exp.pert_begin, exp.pert_end, noise=True)
    # model = TwoHemiRNNTanh_single_readout(configs, exp.a, exp.pert_begin, exp.pert_end)
    exp.init_sub_path(configs['train_type'])

    model_path = os.path.join(exp.configs['models_dir'], exp.configs['model_type'], exp.sub_path)

    # else:
            
    #     model_path = 'dual_alm_rnn_models/{}/{}/n_neurons_256_random_seed_{}/n_epochs_30_n_epochs_across_hemi_0/lr_1.0e-04_bs_256/sigma_input_noise_0.10_sigma_rec_noise_0.10/xs_left_alm_amp_{:.2f}_right_alm_amp_{:.2f}/init_cross_hemi_rel_factor_0.20/'.format(
    #             configs['model_type'],
    #             configs['train_type'],
    #             configs['random_seed'],
    #             float(configs['xs_left_alm_amp']),
    #             float(configs['xs_right_alm_amp'])
            # )
    print(f"Using model: {model_path}")
    
    # Parse CLI args first
    args = parse_args()
    
    # Create visualization
    os.makedirs('figs', exist_ok=True)
    mode = 'last'
    if args.best:
        mode = 'best'
    elif args.epoch is not None:
        mode = f'epoch_{args.epoch}'
    # Filename reflects mode and that this is for large models
    save_path = 'figs/large_rnn_L{}_R{}_weights_visualization_{}_{}_seed{}.pdf'.format(
        configs['xs_left_alm_amp'], configs['xs_right_alm_amp'], configs['train_type'], mode, configs['random_seed'])


    if args.best:
        weights = visualize_rnn_weights(model_path, configs, 'best', save_path=save_path)
    elif args.epoch is not None:
        weights = visualize_rnn_weights(model_path, configs, f'epoch_{args.epoch}', save_path=save_path)
    else:
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
