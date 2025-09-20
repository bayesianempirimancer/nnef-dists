#!/usr/bin/env python3
"""
List all available model training scripts and their configurations.

This script provides an overview of all standardized model training scripts
and their default configurations.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import list_available_configs, get_config


def main():
    """List all available models and configurations."""
    
    print("🚀 AVAILABLE NEURAL NETWORK MODELS")
    print("=" * 60)
    
    # List model training scripts
    models_dir = Path("scripts/training")
    
    if models_dir.exists():
        training_scripts = list(models_dir.glob("train_*.py"))
        
        print(f"\n📋 AVAILABLE MODEL TRAINING SCRIPTS ({len(training_scripts)}):")
        print("-" * 50)
        
        for script in sorted(training_scripts):
            model_name = script.stem.replace("train_", "").replace("_", " ").title()
            print(f"  • {model_name:<25} → {script}")
        
        print(f"\n💡 Usage:")
        print(f"  python scripts/training/train_<model_name>.py")
    
    # List predefined configurations
    print(f"\n⚙️  PREDEFINED CONFIGURATIONS:")
    print("-" * 30)
    
    available_configs = list_available_configs()
    for config_name in available_configs:
        try:
            config = get_config(config_name)
            hidden_sizes = config.network.hidden_sizes
            
            if hidden_sizes:
                arch_desc = f"{len(hidden_sizes)} layers x {hidden_sizes[0]} units"
                if len(set(hidden_sizes)) > 1:
                    arch_desc += f" (varies: {min(hidden_sizes)}-{max(hidden_sizes)})"
            else:
                arch_desc = "Variable architecture"
            
            print(f"  • {config_name:<15} → {arch_desc}")
            print(f"    LR: {config.training.learning_rate}, Batch: {config.training.batch_size}")
            
        except Exception as e:
            print(f"  • {config_name:<15} → Error loading config: {e}")
    
    # List experiment scripts
    experiments_dir = Path("scripts/experiments")
    
    if experiments_dir.exists():
        experiment_scripts = list(experiments_dir.glob("*.py"))
        
        print(f"\n🧪 AVAILABLE EXPERIMENT SCRIPTS ({len(experiment_scripts)}):")
        print("-" * 50)
        
        for script in sorted(experiment_scripts):
            experiment_name = script.stem.replace("_", " ").title()
            print(f"  • {experiment_name:<30} → {script}")
    
    # Directory structure
    print(f"\n📁 DIRECTORY STRUCTURE:")
    print("-" * 25)
    print("  scripts/")
    print("  ├── training/         # Individual model training")
    print("  ├── experiments/      # Comprehensive comparisons")
    print("  ├── archive/          # Old scripts (reference)")
    print("  └── README.md         # Documentation")
    print("")
    print("  src/")
    print("  ├── models/           # Standardized model implementations")
    print("  ├── config.py         # Configuration system")
    print("  ├── base_model.py     # Base classes")
    print("  └── data_utils.py     # Data loading utilities")
    print("")
    print("  plotting/")
    print("  └── model_comparison.py  # Plotting utilities")
    
    print(f"\n✅ Use any of the above scripts to train and evaluate models!")
    print(f"📖 See scripts/README.md for detailed documentation")


if __name__ == "__main__":
    main()
