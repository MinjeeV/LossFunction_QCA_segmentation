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

This repository contains multiple branches for different loss function configurations:

- **`main`**: Original baseline code
- **`clDice_QCA_segmentation`**: Fixed ratio clDice implementation
- **`clDice_dynamic_ratio`**: Dynamic ratio clDice with progressive scheduling

To switch between branches:
```bash
git checkout <branch-name>
```

## Running Experiments

### Using Dynamic Ratio clDice

**Branch: `clDice_dynamic_ratio`**

To run experiments with progressively changing loss weights that emphasize topology preservation over training:
```bash
git checkout clDice_dynamic_ratio
python3 initial.py --use_cldice
```

This implementation uses a three-phase training schedule:

#### Training Schedule
* **Epoch 0-50**: Dice 70%, clDice 30% (fixed)
  - Focus on learning basic vessel shape and segmentation
* **Epoch 51-90**: Dice 70%→10%, clDice 30%→90% (linear transition)
  - Progressively strengthen topology learning
* **Epoch 91-100**: Dice 10%, clDice 90% (fixed)
  - Maximize vessel connectivity and centerline preservation

The dynamic weighting strategy allows the model to first establish solid segmentation boundaries, then gradually shift focus to maintaining vessel topology and connectivity throughout the vascular tree.
