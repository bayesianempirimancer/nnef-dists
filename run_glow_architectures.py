#!/usr/bin/env python3
"""
Script to run Glow training with different architectures for comparison.
"""

import subprocess
import sys
from pathlib import Path

def run_glow_training(num_flow_layers, flow_hidden_size, epochs=50):
    """Run Glow training with specific architecture."""
    print(f"\n{'='*60}")
    print(f"🌊 RUNNING GLOW TRAINING")
    print(f"Architecture: {num_flow_layers} layers x {flow_hidden_size} units")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}")
    
    # Create custom config by modifying the glow config
    config_modifications = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_glow_config

# Get base glow config
config = get_glow_config()

# Modify architecture
config.model_specific.num_flow_layers = {num_flow_layers}
config.model_specific.flow_hidden_size = {flow_hidden_size}

# Set experiment name to include architecture
config.experiment.experiment_name = f"glow_{num_flow_layers}L_{flow_hidden_size}H"

print(f"Using custom Glow config: {{config.model_specific.num_flow_layers}} layers x {{config.model_specific.flow_hidden_size}} units")
"""
    
    # Write temporary config file
    temp_config_file = Path("temp_glow_config.py")
    with open(temp_config_file, 'w') as f:
        f.write(config_modifications)
    
    try:
        # Run the training script
        cmd = [
            "python", "scripts/training/train_glow_ET.py",
            "--epochs", str(epochs)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/jebeck/GitHub/nnef-dists")
        
        if result.returncode == 0:
            print(f"✅ Training completed successfully!")
            print(f"Architecture: {num_flow_layers} layers x {flow_hidden_size} units")
        else:
            print(f"❌ Training failed!")
            print(f"Error: {result.stderr}")
            
    finally:
        # Clean up temporary file
        if temp_config_file.exists():
            temp_config_file.unlink()

def main():
    """Run Glow training with different architectures."""
    
    # Define the three architectures to test
    architectures = [
        (20, 32, "20 layers with 32 units per layer"),
        (10, 64, "10 layers with 64 units per layer"), 
        (10, 32, "10 layers with 32 units per layer")
    ]
    
    print("🚀 Starting Glow Architecture Comparison")
    print("=" * 60)
    
    for num_flow_layers, flow_hidden_size, description in architectures:
        print(f"\n📊 Testing: {description}")
        run_glow_training(num_flow_layers, flow_hidden_size, epochs=50)
    
    print(f"\n{'='*60}")
    print("🎉 All Glow architecture tests completed!")
    print("Check artifacts/ET_models/glow_ET/ for results")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

