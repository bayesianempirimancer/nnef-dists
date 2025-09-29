# Layers Directory

This directory contains the core layer implementations for neural network architectures.

## MLP Layers (`mlp.py`)

### Overview

The MLP (Multi-Layer Perceptron) implementation provides two main components:

1. **MLPLayer**: A single layer with activation, normalization, and dropout
2. **MLPBlock**: A true "block" containing multiple MLP layers in sequence

### MLPLayer

A single MLP layer that includes:
- Dense (linear) transformation
- Activation function (default: swish)
- Layer normalization (optional)
- Dropout (optional)

```python
from src.layers.mlp import MLPLayer

# Single layer: 32D -> 64D
mlp_layer = MLPLayer(features=64)
output = mlp_layer(input)  # input: (batch_size, 32) -> output: (batch_size, 64)
```

### MLPBlock

A true "block" containing multiple MLP layers in sequence. Takes a tuple of feature sizes representing the architecture of the block.

```python
from src.layers.mlp import MLPBlock

# Single-layer block (equivalent to MLPLayer)
mlp_block = MLPBlock(features=(64,))
# Architecture: input -> 64D

# Multi-layer block
mlp_block = MLPBlock(features=(64, 64, 64))
# Architecture: input -> 64D -> 64D -> 64D

# Multi-layer block with different sizes
mlp_block = MLPBlock(features=(128, 64, 32))
# Architecture: input -> 128D -> 64D -> 32D
```

### Key Design Principles

1. **Clear Naming**: 
   - `MLPLayer`: Single layer
   - `MLPBlock`: Multiple layers in sequence

2. **Tuple-based Architecture**: 
   - `MLPBlock(features=(64,))` → Single layer block
   - `MLPBlock(features=(64, 64, 64))` → 3-layer block
   - `MLPBlock(features=(128, 64, 32))` → 3-layer block with different sizes

3. **Flexible Configuration**:
   - All layers support activation, layer normalization, and dropout
   - Consistent interface across single layers and blocks

## ResNet Wrapper (`resnet_wrapper.py`)

### Overview

The ResNet wrapper provides residual connections for any neural network module. It supports both parameter sharing and independent parameters.

### ResNetWrapper (Univariate)

Wraps any module that takes a single input to add residual connections.

```python
from src.layers.resnet_wrapper import ResNetWrapper
from src.layers.mlp import MLPLayer, MLPBlock

# Case 1: ResNet by wrapping a single MLP layer
mlp_layer = MLPLayer(features=64)
resnet_layer = ResNetWrapper(
    base_module=mlp_layer, 
    num_blocks=1,
    share_parameters=False  # Independent parameters (default)
)
# Result: x = x + MLPLayer(x)

# Case 2: ResNet by wrapping a true MLP block (multiple layers)
mlp_block = MLPBlock(features=(64, 64, 64))  # 3 layers: 64 -> 64 -> 64
resnet_block = ResNetWrapper(
    base_module=mlp_block, 
    num_blocks=1,
    share_parameters=False  # Independent parameters (default)
)
# Result: x = x + MLPBlock(x) where MLPBlock has 3 internal layers
```

### Parameter Sharing Control

The `share_parameters` flag controls whether parameters are shared between blocks:

```python
# Independent parameters (default)
resnet = ResNetWrapper(base_module, num_blocks=3, share_parameters=False)
# Creates 3 separate parameter sets: block_0, block_1, block_2

# Shared parameters
resnet = ResNetWrapper(base_module, num_blocks=3, share_parameters=True)
# Reuses the same parameters for all 3 blocks
```

### Dimension Handling

The ResNet wrapper automatically handles dimension mismatches:

```python
# 8D -> 64D case
# Input: 8D, Output: 64D
# ResNet creates: x = MLPBlock(x) + Projection(x)
# Where Projection is a Dense(64) layer that projects 8D -> 64D
```

## Architecture Examples

### Example 1: 5 Independent ResNet Layers

```python
# Configuration for 5 independent ResNet layers
config = MLP_ET_Config(
    hidden_sizes=[64, 64, 64, 64, 64],  # 5 layers
    use_resnet=True,
    num_resnet_blocks=1,                 # Each layer is 1 ResNet block
    share_parameters=False,              # Independent parameters
    embedding_type="default"
)

# Architecture:
# 1. Eta embedding: 2D -> 8D
# 2. ResNet Layer 1: 8D -> 64D (x = x + MLPBlock_1(x))
# 3. ResNet Layer 2: 64D -> 64D (x = x + MLPBlock_2(x))
# 4. ResNet Layer 3: 64D -> 64D (x = x + MLPBlock_3(x))
# 5. ResNet Layer 4: 64D -> 64D (x = x + MLPBlock_4(x))
# 6. ResNet Layer 5: 64D -> 64D (x = x + MLPBlock_5(x))
# 7. Output layer: 64D -> 2D
```

### Example 2: Multi-Layer Blocks with ResNet

```python
# If we wanted true multi-layer blocks:
mlp_block = MLPBlock(features=(64, 64, 64))  # 3 layers inside the block
resnet_block = ResNetWrapper(mlp_block, num_blocks=1)
# Result: x = x + [MLPLayer_1 -> MLPLayer_2 -> MLPLayer_3](x)
```

## Usage in Models

The MLP ET model uses this architecture:

```python
# In MLP_ET_Network.__call__()
for i, hidden_size in enumerate(self.config.hidden_sizes):
    # Create MLP block (single layer)
    mlp_block = MLPBlock(features=(hidden_size,))
    
    # Wrap with ResNet for residual connections
    mlp_resnet = ResNetWrapper(
        base_module=mlp_block,
        num_blocks=self.config.num_resnet_blocks,
        share_parameters=self.config.share_parameters
    )
    
    x = mlp_resnet(x, training=training)
```

## Key Benefits

1. **Intuitive Naming**: Layer vs Block distinction is clear
2. **Flexible Architecture**: Support for both single layers and multi-layer blocks
3. **ResNet Integration**: Easy residual connections with dimension handling
4. **Parameter Control**: Choose between shared and independent parameters
5. **Consistent Interface**: Same API across all layer types

