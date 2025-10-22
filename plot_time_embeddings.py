#!/usr/bin/env python3
"""
Plot time embedding functions for t ∈ [0, 1]
"""

import sys
import os
sys.path.append('/home/jebeck/GitHub/nnef-dists')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from src.embeddings.time_embeddings import (
    SinusoidalTimeEmbedding, 
    CyclicalFourierTimeEmbedding, 
    LogFreqTimeEmbedding,
    LinearTimeEmbedding,
    ConstantTimeEmbedding,
    create_time_embedding
)

def plot_time_embeddings():
    """Plot various time embedding functions for t ∈ [0, 1]"""
    
    # Create time points
    t = jnp.linspace(0, 1, 100)
    embed_dim = 8
    
    # Create embeddings that work properly
    embeddings = {
        'Sinusoidal': SinusoidalTimeEmbedding(embed_dim=embed_dim),
        'Log Frequency': LogFreqTimeEmbedding(min_freq=0.1, max_freq=10.0),
        'Linear': LinearTimeEmbedding(),
        'Constant': ConstantTimeEmbedding()
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Plot each embedding
    for i, (name, embedding) in enumerate(embeddings.items()):
        ax = axes[i]
        
        # Get embeddings for all time points
        embeds = embedding(t)
        
        # Plot each dimension
        for j in range(embed_dim):
            ax.plot(t, embeds[:, j], label=f'Dim {j}', linewidth=2)
        
        ax.set_title(f'{name} Time Embedding', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Embedding Value')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_xlim(0, 1)
    
    # No need to remove subplots since we have exactly 4 embeddings
    
    plt.tight_layout()
    plt.suptitle('Time Embedding Functions for t ∈ [0, 1]', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    output_path = 'time_embeddings_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also create a detailed plot for sinusoidal (most commonly used)
    plt.figure(figsize=(12, 8))
    
    # Test with different embed_dim values for sinusoidal
    embed_dims = [4, 8, 16, 32]
    colors = plt.cm.viridis(np.linspace(0, 1, len(embed_dims)))
    
    for i, dim in enumerate(embed_dims):
        plt.subplot(2, 2, i+1)
        sin_embed = SinusoidalTimeEmbedding(embed_dim=dim)
        embeds = sin_embed(t)
        
        for j in range(dim):
            plt.plot(t, embeds[:, j], label=f'Dim {j}', linewidth=2)
        
        plt.title(f'Sinusoidal Embedding (dim={dim})', fontweight='bold')
        plt.xlabel('Time t')
        plt.ylabel('Embedding Value')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.suptitle('Sinusoidal Time Embedding with Different Dimensions', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    
    # Save the detailed plot
    detailed_output = 'sinusoidal_embeddings_detailed.png'
    plt.savefig(detailed_output, dpi=300, bbox_inches='tight')
    print(f"Detailed plot saved to: {detailed_output}")
    
    plt.show()

if __name__ == "__main__":
    plot_time_embeddings()
