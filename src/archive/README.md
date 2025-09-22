# Archive Directory

This directory contains archived versions of the original unified network classes that have been restructured into individual model files.

## Archived Files

- **`ET_Net.py`** - Original unified ET (Exponential Family) network class that contained all ET architectures (MLP, GLU, Quadratic ResNet, Invertible NN, NoProp-CT, Glow, Geometric Flow)
- **`logZ_Net.py`** - Original unified LogZ (Log Normalizer) network class that contained all LogZ architectures (MLP, GLU, Quadratic ResNet, Convex NN)

## Restructuring

These unified classes have been broken down into individual model files in `src/models/` with the following naming convention:
- `{Architecture}_{Type}_Network` (e.g., `MLP_ET_Network`, `GLU_LogZ_Network`)
- `{Architecture}_{Type}_Trainer` (e.g., `MLP_ET_Trainer`, `GLU_LogZ_Trainer`)

## Benefits of Restructuring

1. **Self-contained models** - Each architecture is now in its own file
2. **Easier maintenance** - Changes to one architecture don't affect others
3. **Better readability** - All related code is in one place
4. **Cleaner imports** - No more complex unified class dependencies
5. **Consistent inheritance** - All models inherit from `BaseNeuralNetwork`
6. **Standardized naming** - All classes follow the same naming convention

## Date Archived

Archived on: $(date)
