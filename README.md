#clDice github README

# Loss Function Experiments for QCA Segmentation

This repository applies clDice loss function to vessel segmentation experiments in coronary angiography (QCA) images.


## 🚀 Installation

### 1. Clone the Repository

**For the modified code with clDice implementation:**

```bash
git clone https://github.com/MinjeeV/LossFunction_QCA_segmentation.git
cd LossFunction_QCA_segmentation
git checkout my-modifications
```

## Running Experiments

### Using Dice Loss Only

If you want to use only the Dice loss function, run:
```bash
python3 initial.py
```

### Using Both Dice and clDice

If you want to run experiments using both Dice and clDice, you can execute the following command:
```bash
python3 initial.py --use_cldice
```

This command runs the code with a fixed ratio between Dice and clDice losses. If you want to modify the ratio, you can change the following line in the code (around line 267):
```python
loss = (loss_DICE * 0.3 + loss_CL * 0.7)
```

Adjust the coefficients (e.g., `0.3` and `0.7`) to set your desired ratio between Dice and clDice.