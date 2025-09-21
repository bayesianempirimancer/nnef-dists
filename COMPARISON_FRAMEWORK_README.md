# Comprehensive Model Comparison Framework

## 🎯 Overview

This framework provides a comprehensive, fair comparison of all neural network approaches for learning exponential family mappings, with detailed analysis of both **performance metrics** and **computational costs**.

## 📊 What Gets Measured

### Performance Metrics
- **Test MSE**: Mean squared error on held-out test set
- **Test MAE**: Mean absolute error on held-out test set
- **Training convergence**: Loss curves over epochs

### Computational Cost Analysis
- **Training Time**: Wall-clock time to train (seconds)
- **Parameter Count**: Total number of model parameters
- **Inference Speed**: Time per prediction (milliseconds/sample)
- **Memory Usage**: Model size and computational efficiency

## 🏗️ Framework Components

### 1. Standardized Data Generation
```bash
python scripts/generate_comparison_data.py
```
- **2000 training samples**
- **500 validation samples** 
- **500 test samples**
- **3D Gaussian exponential family**
- **Fixed random seed (42)** for reproducibility

### 2. Individual Model Training
```bash
python scripts/training/train_comparison_models.py --model MODEL_NAME
```

**Available Models:**
- **ET Models**: `mlp_ET`, `glu_ET`, `geometric_flow_ET`, `glow_ET`, `quadratic_resnet_ET`
- **LogZ Models**: `mlp_logZ`, `glu_logZ`, `quadratic_resnet_logZ`

**Standardized Configuration:**
- **Epochs**: 20 (configurable)
- **Architecture**: [64, 32, 16] hidden layers
- **Learning Rate**: 1e-3
- **Batch Size**: 32
- **Layer Normalization**: Enabled

### 3. Batch Comparison Runner
```bash
bash scripts/run_all_model_comparison.sh
```
- Trains all models sequentially
- **Identical parameters** for fair comparison
- **Timeout protection** (30 min per model)
- **Detailed logging** for each model
- **Error handling** and recovery

### 4. Comprehensive Analysis
```bash
python scripts/create_comparison_analysis.py
```
- Collects results from all trained models
- **Performance comparison plots**
- **Computational cost analysis**
- **Efficiency rankings**
- **Statistical summaries**

## 📁 Results Organization

```
artifacts/
├── ET_models/
│   ├── mlp_ET/
│   │   ├── mlp_ET_complete_results.pkl
│   │   ├── mlp_ET_summary.json
│   │   └── mlp_ET_results.png
│   ├── geometric_flow_ET/
│   │   └── ... (same structure)
│   └── ...
├── logZ_models/
│   ├── mlp_logZ/
│   │   └── ... (same structure)
│   └── ...
└── comprehensive_comparison/
    ├── comprehensive_model_comparison.png
    ├── model_comparison_table.csv
    ├── model_comparison_table.txt
    └── all_results_summary.json
```

## 📊 Generated Visualizations

### Main Comparison Plot (9-panel figure):
1. **Test MSE Comparison** - Performance ranking
2. **Parameter Count** - Model complexity
3. **Training Time** - Training computational cost  
4. **Inference Speed** - Prediction computational cost
5. **Accuracy vs Model Size** - Parameter efficiency
6. **Training Speed vs Accuracy** - Training efficiency
7. **Inference Speed vs Accuracy** - Prediction efficiency
8. **Training Curves** - Convergence comparison
9. **Overall Efficiency Score** - Combined metric

### Performance Table:
- Sortable CSV with all metrics
- Formatted text summary
- JSON export for further analysis

## 🚀 Quick Start

### Option 1: Run Everything
```bash
# Generate data
python scripts/generate_comparison_data.py

# Run all models (takes time!)
bash scripts/run_all_model_comparison.sh

# Generate comprehensive analysis
python scripts/create_comparison_analysis.py
```

### Option 2: Test Individual Models
```bash
# Test model creation
python scripts/quick_model_test.py

# Train specific models
python scripts/training/train_comparison_models.py --model mlp_ET
python scripts/training/train_comparison_models.py --model geometric_flow_ET
python scripts/training/train_comparison_models.py --model mlp_logZ

# Analyze results
python scripts/create_comparison_analysis.py
```

### Option 3: Custom Comparison
```bash
# Run comprehensive comparison framework
python scripts/run_comprehensive_comparison.py
```

## 🎯 Key Features

### ✅ Fair Comparison
- **Same dataset** for all models
- **Identical training parameters**
- **Same number of epochs**
- **Consistent evaluation metrics**

### ✅ Comprehensive Metrics  
- **Performance**: MSE, MAE, convergence
- **Computational Cost**: Training time, inference speed
- **Model Complexity**: Parameter count, architecture
- **Efficiency**: Combined performance/cost metrics

### ✅ Robust Framework
- **Error handling** and timeout protection
- **Detailed logging** for debugging
- **Automatic result collection**
- **Multiple output formats** (plots, tables, JSON)

### ✅ Extensible Design
- **Easy to add new models**
- **Configurable parameters**
- **Modular components**
- **Standardized interfaces**

## 📈 Expected Outputs

After running the full comparison, you'll have:

1. **📊 Comprehensive Plots** showing all models' performance and computational costs
2. **📋 Performance Tables** ranking models by various metrics  
3. **⏱️ Timing Analysis** of training and inference costs
4. **🏆 Efficiency Rankings** combining accuracy and computational cost
5. **📁 Complete Results** for further analysis and publication

## 🔧 Configuration

All models use identical configurations defined in:
- `scripts/training/train_comparison_models.py`: Individual training
- `scripts/run_comprehensive_comparison.py`: Batch comparison
- `scripts/create_comparison_analysis.py`: Analysis parameters

Modify `STANDARD_CONFIG` to adjust:
- Number of epochs
- Network architecture
- Learning parameters
- Computational measurement settings

## 🎯 Perfect for Research

This framework provides everything needed for:
- **Academic papers**: Rigorous comparison with statistical significance
- **Method validation**: Fair evaluation against existing approaches  
- **Performance analysis**: Detailed computational cost breakdown
- **Ablation studies**: Easy to modify and re-run comparisons

The **Geometric Flow ET Network** is treated as just one approach among many, ensuring unbiased evaluation and clear demonstration of its advantages (or areas for improvement).
