#!/usr/bin/env python3
"""
Test script to demonstrate perturbation evaluation functionality
"""

import os
import numpy as np
import torch
from dual_alm_rnn_exp import DualALMRNNExp

def test_perturbation_evaluation():
    """Test the perturbation evaluation functionality"""
    
    # Initialize the experiment
    exp = DualALMRNNExp()
    
    # Generate dataset if it doesn't exist
    if not os.path.exists(exp.configs['data_dir']):
        print("Generating dataset...")
        exp.generate_dataset()
    
    # Set up device
    use_cuda = bool(exp.configs['use_cuda'])
    if use_cuda and not torch.cuda.is_available():
        use_cuda = False
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Data loading parameters
    if use_cuda:
        params = {'batch_size': exp.configs['bs'], 'shuffle': False, 'num_workers': exp.configs['num_workers'], 
        'pin_memory': bool(exp.configs['pin_memory'])}
    else:
        params = {'batch_size': exp.configs['bs'], 'shuffle': False}
    
    # Load test data
    test_save_path = os.path.join(exp.configs['data_dir'], 'test')
    test_sensory_inputs = np.load(os.path.join(test_save_path, 'sensory_inputs.npy'))
    test_trial_type_labels = np.load(os.path.join(test_save_path, 'trial_type_labels.npy'))
    
    test_set = torch.utils.data.TensorDataset(torch.tensor(test_sensory_inputs), torch.tensor(test_trial_type_labels))
    test_loader = torch.utils.data.DataLoader(test_set, **params)
    
    # Load a trained model (you'll need to have trained one first)
    model_type = exp.configs['model_type']
    model_save_path = os.path.join(exp.configs['models_dir'], model_type, exp.sub_path)
    model_path = os.path.join(model_save_path, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using train_type_modular()")
        return
    
    # Load the model
    import sys
    import dual_alm_rnn_models
    model = getattr(dual_alm_rnn_models, model_type)(exp.configs, exp.a, exp.pert_begin, exp.pert_end).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print("Model loaded successfully!")
    
    # Run perturbation evaluation
    print("Running perturbation evaluation...")
    results = exp.eval_with_perturbations(
        model=model,
        device=device,
        loader=test_loader,
        model_type=model_type,
        n_control=500,  # Limit control trials for speed
        seed=42  # For reproducibility
    )



if __name__ == "__main__":
    test_perturbation_evaluation() 