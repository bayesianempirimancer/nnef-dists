"""
Training utilities for progress tracking and timing.

This module provides utilities for adding progress bars and timing
to training loops across all model types.
"""

import time
from typing import Dict, Any, Optional, Callable
from tqdm import tqdm


class TrainingProgress:
    """Training progress tracker with timing and progress bar."""
    
    def __init__(self, total_epochs: int, description: str = "Training"):
        self.total_epochs = total_epochs
        self.description = description
        self.start_time = None
        self.pbar = None
        
    def __enter__(self):
        """Start training progress tracking."""
        self.start_time = time.time()
        self.pbar = tqdm(range(self.total_epochs), desc=self.description, unit="epoch")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End training progress tracking."""
        if self.pbar:
            self.pbar.close()
            
    def update(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, **kwargs):
        """Update progress bar with current metrics."""
        if self.pbar:
            postfix = {'Train Loss': f'{train_loss:.4f}'}
            if val_loss is not None:
                postfix['Val Loss'] = f'{val_loss:.4f}'
            postfix.update({k: f'{v:.4f}' if isinstance(v, float) else str(v) for k, v in kwargs.items()})
            self.pbar.set_postfix(postfix)
            self.pbar.update(1)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed training time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def finish(self) -> float:
        """Finish training and return total time."""
        total_time = self.get_elapsed_time()
        print(f"\n✓ Training completed in {total_time:.1f}s")
        return total_time


def add_timing_to_history(history: Dict[str, Any], total_time: float) -> Dict[str, Any]:
    """Add training time to history dictionary."""
    history = history.copy()
    history['total_training_time'] = total_time
    return history


def train_with_progress(train_fn: Callable, epochs: int, description: str = "Training", **kwargs):
    """
    Wrapper function to add progress tracking to any training function.
    
    Args:
        train_fn: Training function that yields (epoch, train_loss, val_loss) tuples
        epochs: Total number of epochs
        description: Description for progress bar
        **kwargs: Additional arguments for train_fn
        
    Returns:
        Result from train_fn with added timing information
    """
    with TrainingProgress(epochs, description) as progress:
        result = None
        for epoch_data in train_fn(epochs=epochs, **kwargs):
            if isinstance(epoch_data, tuple) and len(epoch_data) >= 2:
                epoch, train_loss = epoch_data[:2]
                val_loss = epoch_data[2] if len(epoch_data) > 2 else None
                progress.update(epoch, train_loss, val_loss)
            else:
                result = epoch_data
                break
        
        total_time = progress.finish()
        
        # Add timing to result if it's a history dict
        if isinstance(result, tuple) and len(result) == 2:
            params, history = result
            history = add_timing_to_history(history, total_time)
            return params, history
        else:
            return result
