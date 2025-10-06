"""
Simple, working ET trainer implementation.

This module provides a clean, working training implementation that can be used
by scripts in the scripts/ directory. It's based on the working MLP training
logic but made generic for all ET models.
"""

import jax
import jax.numpy as jnp
from jax import random
import optax
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import pickle

from typing import Union
from ..models.mlp_et_net import MLP_ET_Network
from ..configs.mlp_et_config import MLP_ET_Config
from ..configs.base_training_config import BaseTrainingConfig

# Type aliases for supported models and configs
ETModel = MLP_ET_Network
ETConfig = MLP_ET_Config


class BaseETTrainer:
    """
    Base ET trainer that handles the core training logic.
    
    This trainer is designed to be used by scripts in the scripts/ directory
    and provides a clean interface for training any ET model.
    """
    
    @staticmethod
    def create_base_argument_parser(description: str = "ET Training Script") -> argparse.ArgumentParser:
        """
        Create a base argument parser with common arguments for all ET models.
        
        Args:
            description: Description for the argument parser
            
        Returns:
            ArgumentParser with common arguments
        """
        parser = argparse.ArgumentParser(description=description)
        
        # Required arguments
        parser.add_argument("--data", type=str, required=True,
                           help="Path to training data pickle file")
        parser.add_argument("--epochs", type=int, required=True,
                           help="Number of training epochs")
        
        # Optional arguments (no defaults - use config class defaults)
        parser.add_argument("--dropout-epochs", type=int,
                           help="Number of epochs to use dropout (default from config)")
        parser.add_argument("--output-dir", type=str,
                           help="Output directory for results (default: auto-generated)")
        
        # Optimizer arguments
        parser.add_argument("--learning-rate", type=float,
                           help="Learning rate (default from config)")
        parser.add_argument("--batch-size", type=int,
                           help="Batch size (default from config)")
        parser.add_argument("--optimizer", type=str, 
                           choices=["adam", "adamw", "sgd", "rmsprop"],
                           help="Optimizer type (default from config)")
        parser.add_argument("--weight-decay", type=float,
                           help="Weight decay (default from config)")
        parser.add_argument("--beta1", type=float,
                           help="Adam beta1 parameter (default from config)")
        parser.add_argument("--beta2", type=float,
                           help="Adam beta2 parameter (default from config)")
        parser.add_argument("--eps", type=float,
                           help="Adam epsilon parameter (default from config)")
        parser.add_argument("--loss-function", type=str, 
                           choices=["mse", "mae", "huber", "model_specific"],
                           help="Loss function (default from config)")
        parser.add_argument("--l1-reg-weight", type=float,
                           help="L1 regularization weight (default from config)")
        
        # Training control arguments
        parser.add_argument("--use-mini-batching", action="store_true",
                           help="Use mini-batching (default from config)")
        parser.add_argument("--no-mini-batching", action="store_true",
                           help="Disable mini-batching (default from config)")
        parser.add_argument("--random-batch-sampling", action="store_true",
                           help="Use random batch sampling (default from config)")
        parser.add_argument("--sequential-batch-sampling", action="store_true",
                           help="Use sequential batch sampling (default from config)")
        
        # Training monitoring arguments
        parser.add_argument("--eval-steps", type=int,
                           help="Steps between evaluations (default from config)")
        parser.add_argument("--save-steps", type=int,
                           help="Steps between model saves (default from config)")
        parser.add_argument("--early-stopping-patience", type=int,
                           help="Epochs to wait before early stopping (default from config)")
        parser.add_argument("--early-stopping-min-delta", type=float,
                           help="Minimum change to qualify as improvement (default from config)")
        parser.add_argument("--log-frequency", type=int,
                           help="Steps between logging (default from config)")
        parser.add_argument("--random-seed", type=int,
                           help="Random seed for reproducibility (default from config)")
        
        # Plotting arguments
        parser.add_argument("--no-plots", action="store_true",
                           help="Skip generating plots")
        parser.add_argument("--plot-data", type=str,
                           help="Path to data file for plotting (default: same as training data)")
        
        return parser

    @staticmethod
    def create_training_config_from_args(args) -> BaseTrainingConfig:
        """
        Create a training configuration from command line arguments.
        
        This method extracts training-related arguments and creates a BaseTrainingConfig
        with only the explicitly provided parameters, letting config defaults handle the rest.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            BaseTrainingConfig instance
        """
        # Build training config kwargs - only include explicitly provided arguments
        training_kwargs = {}
        
        # Direct mapping (arg_name -> config_name)
        training_attributes = [
            'learning_rate', 'batch_size', 'weight_decay', 'beta1', 'beta2', 'eps',
            'loss_function', 'l1_reg_weight', 'use_mini_batching', 'random_batch_sampling',
            'eval_steps', 'save_steps', 'early_stopping_patience', 'early_stopping_min_delta',
            'log_frequency', 'random_seed', 'optimizer', 'dropout_epochs'
        ]
        
        for attribute in training_attributes:
            if hasattr(args, attribute) and getattr(args, attribute) is not None:
                training_kwargs[attribute] = getattr(args, attribute)
        
        # Create training configuration
        return BaseTrainingConfig(**training_kwargs)
    
    def __init__(self, model: ETModel, config: ETConfig):
        """
        Initialize the trainer.
        
        Args:
            model: The ET model to train
            config: Model configuration
        """
        self.model = model
        self.config = config
        self.rng = random.PRNGKey(42)
    
    def _create_optimized_jit_train_step(self, optimizer, batch_size: int):
        """
        Create an optimized JIT-compiled training step for maximum performance.
        
        This method creates a highly optimized training step that:
        1. Uses aggressive JIT compilation
        2. Optimizes memory access patterns
        3. Minimizes Python overhead
        4. Uses device-specific optimizations
        
        Args:
            optimizer: The optimizer to use
            batch_size: Batch size for optimization hints
            
        Returns:
            JIT-compiled training step function
        """
        def optimized_train_step(params, opt_state, eta_batch, mu_T_batch, rng_key, training):
            """
            Optimized training step with comprehensive JIT optimization.
            
            This function is designed for maximum performance with:
            - Single forward pass for loss computation
            - Efficient gradient computation
            - Optimized parameter updates
            - Minimal memory allocations
            """
            def loss_func(params):
                # Create RNGs efficiently - always provide dropout RNG
                rngs = {'dropout': rng_key}
                
                # All models must implement the loss method
                if not hasattr(self.model, 'loss'):
                    raise NotImplementedError(f"Model {type(self.model).__name__} must implement the 'loss' method")
                
                loss = self.model.loss(params, eta_batch, mu_T_batch, training=training, rngs=rngs)
                
                # Add L1 regularization if specified
                l1_weight = getattr(self.config, 'l1_reg_weight', 0.0)
                if l1_weight > 0:
                    l1_loss = jnp.sum(jnp.abs(jax.tree_leaves(params)[0]))  # Simplified L1
                    loss += l1_weight * l1_loss
                
                return loss
            
            # Compute loss and gradients in a single pass for efficiency
            loss, grads = jax.value_and_grad(loss_func)(params)
            
            # Apply optimizer updates
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            return params, opt_state, loss
        
        # Create highly optimized JIT function
        # Use static_argnums for training parameter to enable better optimization
        jit_train_step = jax.jit(
            optimized_train_step,
            static_argnums=(5,),  # training parameter
            device=jax.devices()[0] if jax.devices() else None,
            # Additional JIT optimizations
            inline=True,  # Inline small functions for better performance
        )
        
        return jit_train_step

    def _warm_up_jit_compilation(self, train_step, params, opt_state, eta_sample, mu_T_sample, rng_key):
        """
        Warm up JIT compilation to avoid timing compilation overhead in training.
        
        Args:
            train_step: JIT-compiled training step function
            params: Model parameters
            opt_state: Optimizer state
            eta_sample: Sample eta data
            mu_T_sample: Sample mu_T data
            rng_key: Random key for warmup
        """
        print("Warming up JIT compilation...")
        warmup_start = time.time()
        
        # Warm up both training and inference modes
        try:
            # Warm up training mode
            _, _, _ = train_step(params, opt_state, eta_sample, mu_T_sample, rng_key, True)
            # Warm up inference mode
            _, _, _ = train_step(params, opt_state, eta_sample, mu_T_sample, rng_key, False)
            
            warmup_time = time.time() - warmup_start
            print(f"  JIT compilation warmup completed in {warmup_time:.3f} seconds")
        except Exception as e:
            print(f"  JIT warmup failed: {e}")
            print("  Continuing with training (compilation will happen during first step)")

    def _optimize_training_loop(self, train_step, params, opt_state, batch_indices_list, 
                               train_eta, train_mu_T, val_eta, val_mu_T, num_epochs, dropout_epochs):
        """
        Optimized training loop with performance monitoring and optimizations.
        
        Args:
            train_step: JIT-compiled training step function
            params: Model parameters
            opt_state: Optimizer state
            batch_indices_list: Pre-computed batch indices
            train_eta: Training eta data
            train_mu_T: Training mu_T data
            val_eta: Validation eta data
            val_mu_T: Validation mu_T data
            num_epochs: Number of training epochs
            dropout_epochs: Number of epochs with dropout
            
        Returns:
            Tuple of (final_params, final_opt_state, training_history)
        """
        start_time = time.time()
        training_history = []
        
        print(f"Starting optimized training loop for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Determine if dropout should be active
            use_dropout = epoch < dropout_epochs
            
            # Training with efficient mini-batches
            epoch_train_loss = 0.0
            self.rng, epoch_key = random.split(self.rng)
            
            # Use pre-computed batch indices (OPTIMIZED)
            epoch_batches = batch_indices_list[epoch]
            
            epoch_start = time.time()
            
            for batch_idx, batch_indices in enumerate(epoch_batches):
                # Split RNG key for each batch
                epoch_key, batch_key = random.split(epoch_key)
                
                # Get batch data efficiently
                eta_batch = train_eta[batch_indices]
                mu_T_batch = train_mu_T[batch_indices]
                
                # Execute optimized training step
                params, opt_state, loss = train_step(params, opt_state, eta_batch, mu_T_batch, batch_key, use_dropout)
                
                epoch_train_loss += float(loss)
            
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_train_loss / len(epoch_batches)
            
            # Compute validation loss using current parameters
            val_predictions = self.model.predict(params, val_eta)
            val_loss = float(jnp.mean((val_predictions - val_mu_T) ** 2))
            
            # Store training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'use_dropout': use_dropout
            })
            
            # Print progress
            if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs}: loss={avg_epoch_loss:.6f}, time={epoch_time:.3f}s")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.3f} seconds")
        print(f"Average time per epoch: {total_time/num_epochs:.3f} seconds")
        
        return params, opt_state, training_history

    def _create_optimizer(self, learning_rate: float, training_config=None):
        """
        Create sophisticated optimizer based on config parameters.
        
        Args:
            learning_rate: Base learning rate
            training_config: Training configuration (optional, falls back to self.config)
            
        Returns:
            Configured optimizer
        """
        # Use training_config if provided, otherwise fall back to self.config
        config = training_config if training_config is not None else self.config
        
        # Get optimizer parameters from config with defaults
        weight_decay = getattr(config, 'weight_decay', 0.0)
        beta1 = getattr(config, 'beta1', 0.9)  # First moment decay rate
        beta2 = getattr(config, 'beta2', 0.999)  # Second moment decay rate
        epsilon = getattr(config, 'epsilon', 1e-8)  # Numerical stability
        momentum = getattr(config, 'momentum', 0.9)  # For SGD
        nesterov = getattr(config, 'nesterov', False)  # Nesterov momentum
        optimizer_type = getattr(config, 'optimizer', 'adam')  # 'adam', 'adamw', 'sgd', 'rmsprop'
        
        # Create optimizer based on type
        if optimizer_type.lower() == 'adamw' or (optimizer_type.lower() == 'adam' and weight_decay > 0):
            # AdamW with sophisticated parameters
            optimizer = optax.adamw(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                b1=beta1,
                b2=beta2,
                eps=epsilon
            )
        elif optimizer_type.lower() == 'adam':
            # Standard Adam with sophisticated parameters
            optimizer = optax.adam(
                learning_rate=learning_rate,
                b1=beta1,
                b2=beta2,
                eps=epsilon
            )
        elif optimizer_type.lower() == 'sgd':
            # SGD with momentum and Nesterov
            optimizer = optax.sgd(
                learning_rate=learning_rate,
                momentum=momentum,
                nesterov=nesterov
            )
        elif optimizer_type.lower() == 'rmsprop':
            # RMSprop with sophisticated parameters
            decay = getattr(self.config, 'rmsprop_decay', 0.9)
            optimizer = optax.rmsprop(
                learning_rate=learning_rate,
                decay=decay,
                eps=epsilon,
                momentum=momentum
            )
        else:
            # Default to Adam if unknown optimizer type
            print(f"Warning: Unknown optimizer type '{optimizer_type}', using Adam")
            optimizer = optax.adam(
                learning_rate=learning_rate,
                b1=beta1,
                b2=beta2,
                eps=epsilon
            )
        
        return optimizer
    
    def _compute_loss(self, params: Dict, eta: jnp.ndarray, targets: jnp.ndarray, 
                     rngs: dict, training: bool = True) -> float:
        """
        Compute loss efficiently with exactly one forward pass.
        
        This method ensures exactly one forward pass per loss computation:
        1. Model-specific loss functions handle their own forward pass internally
        2. Standard loss functions compute predictions once and use them
        3. Any internal losses must be included in the model-specific loss function
        
        Args:
            params: Model parameters
            eta: Input natural parameters
            targets: Target values
            rngs: Random number generator keys
            training: Whether in training mode
            
        Returns:
            Computed loss value
        """
        loss_function = getattr(self.config, 'loss_function', 'mse')
        loss_alpha = getattr(self.config, 'loss_alpha', 1.0)
        loss_beta = getattr(self.config, 'loss_beta', 0.0)
        
        # All models must implement the loss method
        if not hasattr(self.model, 'loss'):
            raise NotImplementedError(f"Model {type(self.model).__name__} must implement the 'loss' method")
        
        primary_loss = self.model.loss(params, eta, targets, training=training, rngs=rngs)
        
        # Ensure we don't get NaN and return
        primary_loss = jnp.where(jnp.isfinite(primary_loss), primary_loss, 1e6)        
        return primary_loss
    
    def _print_optimizer_config(self, learning_rate: float, training_config=None):
        """
        Print optimizer configuration for debugging.
        
        Args:
            learning_rate: Base learning rate
            training_config: Training configuration (optional)
        """
        # Use training_config if provided, otherwise fall back to self.config
        config = training_config if training_config is not None else self.config
        
        optimizer_type = getattr(config, 'optimizer', 'adam')
        weight_decay = getattr(config, 'weight_decay', 0.0)
        beta1 = getattr(config, 'beta1', 0.9)
        beta2 = getattr(config, 'beta2', 0.999)
        epsilon = getattr(config, 'epsilon', 1e-8)
        momentum = getattr(config, 'momentum', 0.9)
        nesterov = getattr(config, 'nesterov', False)
        
        print(f"Optimizer Configuration:")
        print(f"  Type: {optimizer_type.upper()}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        
        if optimizer_type.lower() in ['adam', 'adamw']:
            print(f"  Beta1 (momentum): {beta1}")
            print(f"  Beta2 (variance): {beta2}")
            print(f"  Epsilon: {epsilon}")
        elif optimizer_type.lower() == 'sgd':
            print(f"  Momentum: {momentum}")
            print(f"  Nesterov: {nesterov}")
        elif optimizer_type.lower() == 'rmsprop':
            decay = getattr(config, 'rmsprop_decay', 0.9)
            print(f"  Decay: {decay}")
            print(f"  Momentum: {momentum}")
            print(f"  Epsilon: {epsilon}")
    
    def _print_loss_config(self):
        """Print loss function configuration for debugging."""
        loss_function = getattr(self.config, 'loss_function', 'mse')
        loss_alpha = getattr(self.config, 'loss_alpha', 1.0)
        loss_beta = getattr(self.config, 'loss_beta', 0.0)
        huber_delta = getattr(self.config, 'huber_delta', 1.0)
        l1_reg_weight = getattr(self.config, 'l1_reg_weight', 0.0)
        
        print(f"Loss Configuration:")
        print(f"  Primary loss: {loss_function.upper()}")
        print(f"  Primary weight (alpha): {loss_alpha}")
        print(f"  Secondary weight (beta): {loss_beta}")
        if loss_function == 'huber':
            print(f"  Huber delta: {huber_delta}")
        if l1_reg_weight > 0:
            print(f"  L1 regularization: {l1_reg_weight}")
        if hasattr(self.model, 'loss'):
            print(f"  Model-specific loss: Available")
        if hasattr(self.model, 'compute_internal_loss'):
            print(f"  Model internal loss: Available")
    
    def _count_parameters(self, params) -> int:
        """
        Count the total number of parameters in the model.
        
        Args:
            params: Model parameters (JAX pytree)
            
        Returns:
            Total number of parameters
        """
        param_count = 0
        for param_tree in jax.tree_util.tree_leaves(params):
            param_count += param_tree.size
        return param_count
    
    def _sample_batch(self, rng_key, data, batch_size, batch_idx, use_random=True):
        """
        Sample a batch from data.
        
        Args:
            rng_key: JAX random key
            data: Data array to sample from
            batch_size: Size of batch to sample
            batch_idx: Current batch index (for sequential sampling)
            use_random: Whether to use random sampling (True) or sequential (False)
            
        Returns:
            Sampled batch
        """
        if use_random:
            # Random sampling with replacement
            indices = random.choice(rng_key, len(data), shape=(batch_size,), replace=True)
            return data[indices]
        else:
            # Sequential sampling with wraparound
            start_idx = (batch_idx * batch_size) % len(data)
            end_idx = start_idx + batch_size
            if end_idx > len(data):
                # Handle wraparound
                indices = jnp.concatenate([
                    jnp.arange(start_idx, len(data)),
                    jnp.arange(0, end_idx - len(data))
                ])
            else:
                indices = jnp.arange(start_idx, end_idx)
            return data[indices]
    
    def _create_optimized_batch_indices(self, n_samples: int, batch_size: int, num_epochs: int, 
                                      rng_key: jax.random.PRNGKey, use_random: bool = True) -> List[List[jnp.ndarray]]:
        """
        Pre-compute all batch indices for all epochs for optimized batch creation.
        
        This optimization is particularly beneficial for larger batch sizes (>=64) where
        the overhead of random permutation becomes significant.
        
        Args:
            n_samples: Total number of training samples
            batch_size: Batch size
            num_epochs: Number of training epochs
            rng_key: Random key for permutation
            use_random: Whether to use random sampling (True) or sequential (False)
            
        Returns:
            List of lists: batch_indices_list[epoch][batch_idx] contains the indices for that batch
        """
        batch_indices_list = []
        
        for epoch in range(num_epochs):
            if use_random:
                # Generate random permutation for this epoch
                epoch_rng = jax.random.fold_in(rng_key, epoch)
                indices = jax.random.permutation(epoch_rng, n_samples)
                
                # Create batches for this epoch
                epoch_batches = []
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    epoch_batches.append(batch_indices)
            else:
                # Sequential batching - much simpler
                epoch_batches = []
                for i in range(0, n_samples, batch_size):
                    batch_indices = jnp.arange(i, min(i + batch_size, n_samples))
                    epoch_batches.append(batch_indices)
            
            batch_indices_list.append(epoch_batches)
        
        return batch_indices_list
    
    
    def _compute_regularization_loss(self, params: Dict) -> float:
        """
        Compute regularization losses based on config.
        
        Args:
            params: Model parameters
            
        Returns:
            Regularization loss value
        """
        total_reg_loss = 0.0
        
        # L1 regularization
        l1_weight = getattr(self.config, 'l1_reg_weight', 0.0)
        if l1_weight > 0:
            l1_loss = 0.0
            for param_tree in jax.tree_leaves(params):
                l1_loss += jnp.sum(jnp.abs(param_tree))
            total_reg_loss += l1_weight * l1_loss
        
        # L2 regularization (weight decay is handled by optimizer)
        # Additional L2 can be added here if needed
        
        return total_reg_loss
    
    def _save_checkpoint(self, params: Dict, opt_state: Any, epoch: int, 
                        train_loss: float, val_loss: float, output_dir: str):
        """
        Save training checkpoint.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            epoch: Current epoch
            train_loss: Current training loss
            val_loss: Current validation loss
            output_dir: Directory to save to
        """
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'params': params,
            'opt_state': opt_state,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config.__dict__
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Dict, Any, int, float, float]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (params, opt_state, epoch, train_loss, val_loss)
        """
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"  Train loss: {checkpoint['train_loss']:.6f}")
        print(f"  Val loss: {checkpoint['val_loss']:.6f}")
        
        return (
            checkpoint['params'],
            checkpoint['opt_state'],
            checkpoint['epoch'],
            checkpoint['train_loss'],
            checkpoint['val_loss']
        )
    
    def train(self, 
              train_eta: jnp.ndarray, 
              train_mu_T: jnp.ndarray,
              val_eta: jnp.ndarray, 
              val_mu_T: jnp.ndarray,
              num_epochs: int,
              dropout_epochs: int = 0,
              learning_rate: Optional[float] = None,
              batch_size: Optional[int] = None,
              eval_steps: int = 10,
              save_steps: Optional[int] = None,
              output_dir: Optional[str] = None,
              training_config: Optional[BaseTrainingConfig] = None) -> Dict[str, Any]:
        """
        Train the model using the provided data with sophisticated config-based parameters.
        
        Args:
            train_eta: Training natural parameters [N_train, eta_dim]
            train_mu_T: Training target statistics [N_train, mu_dim]
            val_eta: Validation natural parameters [N_val, eta_dim]
            val_mu_T: Validation target statistics [N_val, mu_dim]
            num_epochs: Number of training epochs
            dropout_epochs: Number of epochs to use dropout (0 = no dropout)
            learning_rate: Learning rate (uses config default if None)
            batch_size: Batch size for training (uses config default if None)
            eval_steps: Steps between evaluations
            save_steps: Steps between model saves (None = no intermediate saves)
            output_dir: Directory for saving checkpoints (None = no saving)
            
        Returns:
            Dictionary with training results
        """
        # Use config parameters if not provided
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Check if mini-batching should be used
        use_mini_batching = getattr(self.config, 'use_mini_batching', True)
        random_sampling = getattr(self.config, 'random_sampling', True)
        if not use_mini_batching:
            batch_size = len(train_eta)  # Use entire dataset as one batch
            
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training data: {train_eta.shape[0]} samples")
        print(f"Validation data: {val_eta.shape[0]} samples")
        print(f"Learning rate: {learning_rate}")
        print(f"Mini-batching: {'ON' if use_mini_batching else 'OFF'}")
        if use_mini_batching:
            print(f"Batch size: {batch_size}")
            print(f"Sampling: {'RANDOM' if random_sampling else 'SEQUENTIAL'}")
        else:
            print(f"Batch size: {len(train_eta)}")
        print(f"Dropout epochs: {dropout_epochs}")
        print(f"Save checkpoints: {'ON' if save_steps is not None else 'OFF'}")
        if save_steps is not None:
            print(f"Save steps: {save_steps}")
        
        # Initialize model parameters
        self.rng, init_key = random.split(self.rng)
        
        # Handle different model initialization signatures
        if hasattr(self.model, '__call__'):
            # Check if this is a NoProp model (requires z, eta, t)
            import inspect
            sig = inspect.signature(self.model.__call__)
            param_names = list(sig.parameters.keys())
            
            # Check if this is a NoProp model by looking for 'z' parameter
            if 'z' in param_names and len(param_names) >= 3:
                # NoProp model initialization (z, eta, t, training)
                z_sample = jnp.zeros_like(train_eta[:1])
                t_sample = jnp.array(0.5)
                params = self.model.init(init_key, z_sample, train_eta[:1], t_sample, training=True)
            else:
                # Standard model initialization (eta, training)
                params = self.model.init(init_key, train_eta[:1], training=True)
        else:
            # Fallback initialization
            params = self.model.init(init_key, train_eta[:1])
        
        # Create sophisticated optimizer based on config
        optimizer = self._create_optimizer(learning_rate, training_config)
        opt_state = optimizer.init(params)
        
        # Print optimizer and loss configuration
        self._print_optimizer_config(learning_rate, training_config)
        self._print_loss_config()
        
        # Training loop with efficient mini-batching
        train_losses = []
        val_losses = []
        
        # Calculate number of batches for progress tracking
        num_batches = len(train_eta) // batch_size
        if len(train_eta) % batch_size != 0:
            num_batches += 1
        
        # Use class method for batch sampling
        
        # Create optimized JIT training step for maximum performance
        print("Creating optimized JIT training step...")
        train_step = self._create_optimized_jit_train_step(optimizer, batch_size)
        
        # Warm up JIT compilation to avoid timing compilation overhead
        self.rng, warmup_rng = random.split(self.rng)
        self._warm_up_jit_compilation(train_step, params, opt_state, train_eta[:batch_size], train_mu_T[:batch_size], warmup_rng)
        
        # OPTIMIZATION: Pre-compute batch indices for all training
        print(f"Pre-computing batch indices for batch_size={batch_size}")
        prep_start = time.time()
        batch_indices_list = self._create_optimized_batch_indices(
            len(train_eta), batch_size, num_epochs, self.rng, random_sampling
        )
        prep_time = time.time() - prep_start
        print(f"  Batch indices pre-computed in {prep_time:.3f} seconds")
        
        # Use optimized training loop
        start_time = time.time()
        params, opt_state, training_history = self._optimize_training_loop(
            train_step, params, opt_state, batch_indices_list, 
            train_eta, train_mu_T, val_eta, val_mu_T, num_epochs, dropout_epochs
        )
        training_time = time.time() - start_time
        
        # Extract training and validation losses from history
        train_losses = [entry['train_loss'] for entry in training_history]
        val_losses = [entry['val_loss'] for entry in training_history]
        
        # Print progress with validation
        for epoch in range(num_epochs):
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                use_dropout = epoch < dropout_epochs
                dropout_status = "ON" if use_dropout else "OFF"
                print(f"Epoch {epoch:3d}: Train Loss = {train_losses[epoch]:.6f}, Val Loss = {val_losses[epoch]:.6f} (Dropout: {dropout_status})")
            
            # Save checkpoint if requested
            if save_steps is not None and output_dir is not None and epoch % save_steps == 0:
                self._save_checkpoint(params, opt_state, epoch, train_losses[epoch], val_losses[epoch], output_dir)
        
        # Compute final metrics
        final_train_loss = train_losses[-1] if train_losses else 0.0
        final_val_loss = val_losses[-1] if val_losses else 0.0
        best_val_loss = min(val_losses) if val_losses else 0.0
        
        # Compute inference time
        print("\nComputing inference time...")
        test_batch_size = 100
        test_eta = val_eta[:test_batch_size]
        
        # Warm up (JAX compilation) - handle different model signatures
        if hasattr(self.model, '__call__'):
            import inspect
            sig = inspect.signature(self.model.__call__)
            param_names = list(sig.parameters.keys())
            
            # Check if this is a NoProp model by looking for 'z' parameter
            if 'z' in param_names and len(param_names) >= 3:
                # NoProp model inference
                z_sample = jnp.zeros_like(test_eta[:1])
                t_sample = jnp.array(0.5)
                _ = self.model.apply(params, z_sample, test_eta[:1], t_sample, training=False)
            else:
                # Standard model inference
                _ = self.model.apply(params, test_eta[:1], training=False)
        else:
            # Fallback inference
            _ = self.model.apply(params, test_eta[:1], training=False)
        
        # Time inference
        inference_times = []
        for _ in range(10):
            start_inference = time.time()
            # Handle different model signatures for timing
            if hasattr(self.model, '__call__'):
                import inspect
                sig = inspect.signature(self.model.__call__)
                param_names = list(sig.parameters.keys())
                
                # Check if this is a NoProp model by looking for 'z' parameter
                if 'z' in param_names and len(param_names) >= 3:
                    # NoProp model inference
                    z_sample = jnp.zeros_like(test_eta)
                    t_sample = jnp.array(0.5)
                    _ = self.model.apply(params, z_sample, test_eta, t_sample, training=False)
                else:
                    # Standard model inference
                    _ = self.model.apply(params, test_eta, training=False)
            else:
                # Fallback inference
                _ = self.model.apply(params, test_eta, training=False)
            inference_times.append(time.time() - start_inference)
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        inference_time_per_sample = avg_inference_time / test_batch_size
        
        print(f"  Inference time: {avg_inference_time:.6f} seconds for batch of {test_batch_size}")
        print(f"  Inference time per sample: {inference_time_per_sample:.6f} seconds")
        
        # Count model parameters
        param_count = self._count_parameters(params)
        print(f"  Model parameters: {param_count:,}")
        
        # Create results dictionary
        results = {
            'params': params,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'inference_time': avg_inference_time,
            'inference_time_per_sample': inference_time_per_sample,
            'inference_batch_size': test_batch_size,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config.__dict__,
            'parameter_count': param_count
        }
        
        print(f"\nTraining Results:")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Final train loss: {final_train_loss:.6f}")
        print(f"  Final val loss: {final_val_loss:.6f}")
        print(f"  Best val loss: {best_val_loss:.6f}")
        
        return results
    
    def save_model(self, output_dir: str, results: Dict[str, Any]):
        """
        Save the trained model and results.
        
        Args:
            output_dir: Directory to save to
            results: Training results dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training results
        with open(output_path / "training_results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        # Save model config
        if hasattr(self.config, 'save_pretrained'):
            self.config.save_pretrained(str(output_path))
        else:
            # Fallback: save as JSON
            import json
            with open(output_path / "config.json", "w") as f:
                json.dump(results['config'], f, indent=2)
        
        # Save model parameters
        with open(output_path / "model_params.pkl", "wb") as f:
            pickle.dump(results['params'], f)
        
        print(f"Training results saved to: {output_path / 'training_results.pkl'}")
        print(f"Model config saved to: {output_path / 'config.json'}")
        print(f"Model parameters saved to: {output_path / 'model_params.pkl'}")
        print("✅ Training complete!")
