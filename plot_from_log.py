import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves_from_log(log_file, save_path='training_curves.png'):
    """
    로그 파일을 읽어서 학습 곡선 시각화
    """
    # CSV 로그 파일 읽기
    df = pd.read_csv(log_file)
    
    # N/A를 NaN으로 변환
    df['Val_Loss'] = pd.to_numeric(df['Val_Loss'], errors='coerce')
    df['clDice_Score'] = pd.to_numeric(df['clDice_Score'], errors='coerce')
    df['Combined_Score'] = pd.to_numeric(df['Combined_Score'], errors='coerce')
    # std에 대한건 처리x(std관련 그래프 없음)
    
    epochs = df['Epoch'].values
    train_losses = df['Train_Loss'].values
    val_losses = df['Val_Loss'].values
    dice_scores = df['Dice_Score'].values
    cldice_scores = df['clDice_Score'].values
    combined_scores = df['Combined_Score'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Loss curves
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2, 
                    marker='o', markersize=4, markevery=1)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Three validation scores comparison
    axes[0, 1].plot(epochs, dice_scores, label='Dice Score', color='green', 
                    linewidth=2, marker='s', markersize=3, markevery=5)
    axes[0, 1].plot(epochs, cldice_scores, label='clDice Score', color='orange', 
                    linewidth=2, marker='^', markersize=3, markevery=5)
    axes[0, 1].plot(epochs, combined_scores, label='Combined Score (0.5*Dice + 0.5*clDice)', 
                    color='purple', linewidth=2.5, marker='o', markersize=3, markevery=5)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Validation Scores Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss gap (Overfitting indicator)
    valid_idx = ~np.isnan(val_losses)
    if np.any(valid_idx):
        loss_gap = val_losses[valid_idx] - train_losses[valid_idx]
        gap_epochs = epochs[valid_idx]
        
        axes[1, 0].plot(gap_epochs, loss_gap, label='Val Loss - Train Loss', 
                       color='purple', linewidth=2, marker='o', markersize=4)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss Gap', fontsize=12)
        axes[1, 0].set_title('Overfitting Indicator (Gap > 0 → Overfitting)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Score difference (Dice vs clDice)
    score_diff = dice_scores - cldice_scores
    axes[1, 1].plot(epochs, score_diff, label='Dice - clDice', 
                   color='teal', linewidth=2, marker='d', markersize=3, markevery=5)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score Difference', fontsize=12)
    axes[1, 1].set_title('Dice vs clDice Difference (>0: Dice better, <0: clDice better)', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n{'='*60}")
    print(f"Training curves saved: {save_path}")
    print(f"{'='*60}")