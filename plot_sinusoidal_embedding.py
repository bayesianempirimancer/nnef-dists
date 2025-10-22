#!/usr/bin/env python3
"""
Plot sinusoidal time embedding function for t ∈ [0, 1]
This is the main time embedding used in the NoProp CT model.
"""

import sys
import os
sys.path.append('/home/jebeck/GitHub/nnef-dists')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from src.embeddings.time_embeddings import SinusoidalTimeEmbedding

def plot_sinusoidal_embedding():
    """Plot sinusoidal time embedding function for t ∈ [0, 1]"""
    
    # Create time points
    t = jnp.linspace(0, 1, 100)
    
    # Create figure with subplots for different embedding dimensions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Test with different embed_dim values
    embed_dims = [4, 8, 16, 32]
    
    for i, dim in enumerate(embed_dims):
        ax = axes[i//2, i%2]
        
        # Create sinusoidal embedding
        sin_embed = SinusoidalTimeEmbedding(embed_dim=dim)
        embeds = sin_embed(t)
        
        # Plot each dimension
        for j in range(dim):
            ax.plot(t, embeds[:, j], label=f'Dim {j}', linewidth=2)
        
        ax.set_title(f'Sinusoidal Time Embedding (dim={dim})', fontweight='bold')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Embedding Value')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Sinusoidal Time Embedding Functions for t ∈ [0, 1]', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    output_path = 'sinusoidal_time_embeddings.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Create a detailed plot showing the frequency components
    plt.figure(figsize=(10, 6))
    
    # Use 8-dimensional embedding for detailed analysis
    dim = 8
    sin_embed = SinusoidalTimeEmbedding(embed_dim=dim)
    embeds = sin_embed(t)
    
    # Plot each dimension with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, dim))
    
    for j in range(dim):
        plt.plot(t, embeds[:, j], label=f'Dimension {j}', linewidth=2, color=colors[j])
    
    plt.title(f'Detailed Sinusoidal Time Embedding (dim={dim})', fontsize=14, fontweight='bold')
    plt.xlabel('Time t', fontsize=12)
    plt.ylabel('Embedding Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, 1)
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save the detailed plot
    detailed_output = 'sinusoidal_embedding_detailed.png'
    plt.savefig(detailed_output, dpi=300, bbox_inches='tight')
    print(f"Detailed plot saved to: {detailed_output}")
    
    # Print some information about the embedding
    print(f"\nSinusoidal Time Embedding Analysis:")
    print(f"Embedding dimension: {dim}")
    print(f"Time range: [0, 1]")
    print(f"Number of frequency components: {dim//2}")
    print(f"Frequency range: log-space from 1 to 10000")
    
    # Show the actual frequencies used
    half = dim // 2
    log_freqs = -jnp.log(10000) * jnp.linspace(0, 1, half)
    freqs = jnp.exp(log_freqs)
    print(f"Actual frequencies used: {freqs}")
    
    plt.show()

if __name__ == "__main__":
    plot_sinusoidal_embedding()
