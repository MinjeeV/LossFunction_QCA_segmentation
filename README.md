# Loss Function Experiments for QCA Segmentation

This repository applies clDice loss function to vessel segmentation experiments in coronary angiography (QCA) images.


## 🚀 Installation

### 1. Clone the Repository

**For the modified code with clDice implementation:**
```bash
git clone https://github.com/MinjeeV/LossFunction_QCA_segmentation.git
cd LossFunction_QCA_segmentation
```

## Branch Overview

This repository contains multiple branches for different loss function configurations and evaluation approaches:

- **`main`**: Original baseline code
- **`clDice_QCA_segmentation`**: Fixed ratio clDice implementation
- **`clDice_dynamic_ratio`**: Dynamic ratio clDice with progressive scheduling
- **`add-evaluation-metrics`**: Enhanced evaluation with comprehensive metrics (based on fixed ratio code)

To switch between branches:
```bash
git checkout <branch-name>
```


## Running Experiments

## 📊 Enhanced Evaluation Metrics

**Branch: `add-evaluation-metrics`**

### Overview

Previous evaluations relied primarily on a combined score:
```python
Combined Score = 0.5 × DICE + 0.5 × clDice
```

While this metric provided insights into overall performance trends, it had limitations in capturing the nuanced characteristics of model behavior. The `add-evaluation-metrics` branch addresses this by implementing comprehensive evaluation metrics for detailed performance analysis.

**Note:** This branch is based on the fixed ratio clDice implementation from `clDice_QCA_segmentation`.

### Features

This branch introduces:

1. **Detailed Metric Logging**: Extended evaluation metrics saved during training
2. **Automated Visualization**: Training plots generated automatically after training completion
3. **Modular Testing**: Separated testing functionality for cleaner code structure
4. **Flexible Loss Weighting**: Configure Dice and clDice ratios via command-line arguments

### Usage

### Complete Workflow Example
```bash
# 1. Switch to evaluation metrics branch
git checkout add-evaluation-metrics

# 2. Train model with custom loss weights
# (plots will be automatically generated after training)
python3 initial.py --use_cldice --param_name exp_dice30_cldice70 --dice_w 0.3 --cldice_w 0.7

# 3. Test the trained model
python3 test_model.py --model_name U2net \
                      --model_file exp_dice30_cldice70_epoch100_batch16_fold0 \
                      --param_path ./param/
```

#### Training with Enhanced Metrics
```bash
git checkout add-evaluation-metrics

# Basic usage (REQUIRED: must specify param_name)
python3 initial.py --use_cldice --param_name experiment_name

# Custom loss weights (default: dice_w=0.5, cldice_w=0.5)
python3 initial.py --use_cldice --param_name experiment_name --dice_w 0.3 --cldice_w 0.7

# Example: Emphasize topology preservation
python3 initial.py --use_cldice --param_name topology_focus --dice_w 0.2 --cldice_w 0.8
```

**Required Arguments:**
- `--use_cldice`: Enable clDice loss
- `--param_name`: Experiment identifier (used for saving logs and checkpoints)

**Optional Arguments:**
- `--dice_w`: Weight for Dice loss (default: 0.5)
- `--cldice_w`: Weight for clDice loss (default: 0.5)

**Note:** After training completes, `plot_from_log.py` is automatically executed to generate visualization plots from the training logs.

#### Testing Trained Models
```bash
python3 test_model.py --model_name U2net \
                      --model_file zerotest_epoch100_batch16_fold0 \
                      --param_path ./param/
```

**Required Arguments:**
- `--model_name`: Model architecture (e.g., U2net, UnetPlusPlus, DeepLabV3Plus)
- `--model_file`: Name of the trained model checkpoint (without extension)
- `--param_path`: Directory containing model parameters

### File Structure

#### Core Files

- **`initial.py`**: Main training script with enhanced metric logging
  - Tracks individual loss components (Dice, clDice)
  - Records detailed performance metrics per epoch
  - Saves logs to `train_log/` directory
  - Supports configurable loss weights via command-line arguments
  - Automatically calls `plot_from_log.py` after training completion

- **`test_model.py`**: Standalone testing module
  - Separated from training pipeline for better code organization
  - Evaluates trained models on test datasets
  - Provides comprehensive performance reports

- **`plot_from_log.py`**: Visualization generator
  - Called automatically by `initial.py` after training
  - Reads metrics from `train_log/` directory
  - Generates training curves and performance plots
  - Visualizes loss components separately for detailed analysis

#### Output Directory

- **`train_log/`**: Contains all training logs and generated visualizations
  - Metric logs (CSV/text format)
  - Training curves (automatically generated)
  - Loss component plots (automatically generated)
  - Organized by `param_name`




## 📁 Directory Structure
```
LossFunction_QCA_segmentation/
├── initial.py              # Main training script
├── test_model.py          # Testing module (add-evaluation-metrics branch)
├── plot_from_log.py       # Visualization tool (called by initial.py)
├── train_log/             # Training logs and plots (generated)
│   ├── <param_name>/
│   │   ├── metrics.csv
│   │   └── *.png
├── param/                 # Saved model checkpoints
│   └── <param_name>/
├── models/                # Model architectures
├── utils/                 # Utility functions
└── README.md
```

## 📝 Notes

- All branches maintain compatibility with the same dataset format
- Checkpoints are saved automatically during training with the specified `param_name`
- The `add-evaluation-metrics` branch requires `--param_name` argument for proper experiment tracking
- Visualization plots are automatically generated at the end of training - no manual plotting required
- Loss weights (--dice_w, --cldice_w) should sum to reasonable values for balanced training
- For detailed loss function implementation, refer to the code comments in each branch
