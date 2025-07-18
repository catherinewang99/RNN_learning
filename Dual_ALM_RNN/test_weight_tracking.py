#!/usr/bin/env python3
"""
Test script to demonstrate weight tracking functionality
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dual_alm_rnn_exp import DualALMRNNExp

def test_weight_tracking():
    """Test the weight tracking functionality"""
    
    # Initialize the experiment
    exp = DualALMRNNExp()
    
    # Generate dataset if it doesn't exist
    if not os.path.exists(exp.configs['data_dir']):
        print("Generating dataset...")
        exp.generate_dataset()
    
    # Run modular training (this will automatically track weights)
    print("Starting modular training with weight tracking...")
    exp.train_type_modular()
    
    print("Training complete! Check the logs directory for weight files and plots directory for analysis plots.")

if __name__ == "__main__":
    test_weight_tracking() 