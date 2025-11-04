import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.v2 import GaussianNoise

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler, autocast
from torch import amp

import segmentation_models_pytorch as smp
import albumentations as album

import utils as U

import U2net
import cv2
import numpy as np
import time, os
import json
import argparse

from plot_from_log import plot_training_curves_from_log

from clDice.cldice_loss.pytorch.cldice import soft_dice_cldice, soft_cldice 


def to_float32(x):
    """Convert numpy array to float32 dtype"""
    return x.astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 모델명 필수입력
    parser.add_argument('--param_name', type=str, required=True,
                   help='The model parameter will be saved as this name.')
    # combined score 기준 best score을 가진 모델이 해당 dir에 저장됨.
    parser.add_argument('--param_path', type=str, default='./param/',
                       help='The model parameter will be saved at this path.')
    parser.add_argument('--project', type=str, default='QCA_segmentation',
                       help='The model parameter will be saved at this path.')
    
    parser.add_argument('--model', choices=['U2net', 'Deeplab', 'UnetPP'],
                        default='U2net')
    parser.add_argument('--dataset', type=str, default='CBN')
    parser.add_argument('--fold', type=int, default=0)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--use_fp16', type=int, default=0,
                help='Use mixed precision method: set a number of iters_to_accumulate')
    
    # result_log 저장 파라미터
    parser.add_argument('--train_log_path', type=str, default='./train_log/',
                       help='The model parameter will be saved at this path.')
    
    # clDice 관련 파라미터
    parser.add_argument('--use_cldice', action='store_true',
                       help='Use clDice loss for topology preservation')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for clDice loss (0.0=only dice, 1.0=only cldice)')
    parser.add_argument('--skel_iter', type=int, default=10,
                       help='Number of iterations for soft skeletonization')
    
    # loss ratio
    parser.add_argument('--dice_w', type=float, default=0.5)
    parser.add_argument('--cldice_w', type=float, default=0.5)
    
    args = parser.parse_args()
    return args

def select_aug(aug_list):
    transforms = []
    for i in aug_list:
        if i == 'Translation':
            transforms.append(
                album.Affine(
                    scale=(0.8, 1.0),
                    rotate=(-20, 20),
                    translate_percent=(0.0, 0.1),
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(77, 102, 127),
                    p=1
                )
            )

        elif i == 'Contrast':
            transforms.append(album.RandomContrast(limit=0.4, p=0.5))

        elif i == 'GaussNoise':
            transforms.append(album.GaussNoise(std_range=(0, 0.1), p=0.5))

        elif i == 'GaussBlur':
            transforms.append(album.GaussianBlur(blur_limit=5, p=0.2))

        elif i == 'Gamma':
            transforms.append(album.RandomGamma(gamma_limit=(50,150), p=0.2))

        else:
            print('name error')
            return None

    return album.Compose(transforms=transforms)

class QCAdataset(Dataset):
    def __init__(self, idxs, meta, augment=False, dataset='CBN'):
        self.idxs = idxs
        self.meta = meta
        self.base_path = '/CBN/yi/QCA/10. DatasetForDL/'+dataset
        self.augment = augment
              
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        _path = os.path.join(self.base_path, 
                                'Input', 'png', 
                             self.meta[self.idxs[idx]]['id']+'.png')
        img = cv2.imread(_path)[:,:,0]

        _path = os.path.join(self.base_path, 
                                'Mask', 'extracted', 
                            self.meta[self.idxs[idx]]['id']+'.png')
        mask = cv2.imread(_path)
        
        # 채널 3번(index 2)만 선택
        mask = mask[:, :, 2:3]  # (H, W, 1)

        mm = np.min(img)
        image = to_float32((img - mm) / (np.max(img) - mm))
        mask  = to_float32(mask / 255)
        
        if self.augment:
            augmented = self.augment(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        else:
            image, mask = image, mask

        sample = {
            'image': image,
            'mask': np.transpose(mask, (2,0,1)),  # (H,W,1) → (1,H,W)
        }
        return sample

def validation(model, args, data_loader, thres_prop=False, compute_loss=True):
    """
    Validation with optional loss tracking
    Args:
        compute_loss: validation loss를 계산할지 여부 (시간 절약용)
    """
    dice_scores = []
    cl_dice_scores = []
    val_losses = [] if compute_loss else None
    
    model.eval()
    
    for i, data in enumerate(data_loader):
        _img, _mask = data['image'], data['mask']
        _img = _img.float().to(args.device)
        _img = _img.unsqueeze(1)
        _mask_np = _mask.cpu().detach().numpy()

        with torch.no_grad():
            _out = model(_img)
            pred_seg = _out
        
        # U2Net의 경우 main output 추출
        if args.model == 'U2net':
            pred_seg = pred_seg[0]
        
        
        # GPU 마스크 준비 (loss 계산 또는 clDice 사용 시 필요)
        # ═══════════════════════════════════════
        need_mask_gpu = compute_loss or (hasattr(args, 'use_cldice') and args.use_cldice)
        if need_mask_gpu:
            _mask_gpu = _mask.float().to(args.device)
        
        # Validation Loss 계산 (optional)
        # ═══════════════════════════════════════
        if compute_loss:
            if args.model == 'U2net':
                loss_DICE = args.criterion(_out, _mask_gpu)
            else:
                loss_DICE = args.criterion(pred_seg, _mask_gpu)
            
            if hasattr(args, 'use_cldice') and args.use_cldice:
                loss_CL = args.criterion_cl(pred_seg, _mask_gpu)
                
                w_dice = args.dice_w
                w_cl = args.cldice_w
                
                total_loss = w_dice * loss_DICE + w_cl * loss_CL
            else:
                total_loss = loss_DICE
            
            val_losses.append(total_loss.item())
        # ═══════════════════════════════════════
                    
        # Dice score 계산 (기존)
        pred_binary = (pred_seg>0.5).type(torch.uint8).cpu().detach().numpy()
            
        if thres_prop != False:
            pred_binary = U.post_processing(pred_binary, thres_prop=thres_prop)
            
        # 단일 채널 Dice score 계산
        Score = U.calculate_score(_mask_np[:,0], pred_binary[:,0], 'Dice')
        dice_scores.extend(Score)
        
        # clDice score 계산 (topology 평가)
        if hasattr(args, 'use_cldice') and args.use_cldice:
            with torch.no_grad():
                cl_loss = args.criterion_cl(pred_seg, _mask_gpu).item()
                cl_score = 1.0 - cl_loss
                cl_dice_scores.append(cl_score)
        
    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    # val_loss 처리
    if compute_loss and val_losses:
        avg_val_loss = np.mean(val_losses)
    else:
        avg_val_loss = None
    
    # Combined metric 계산
    if cl_dice_scores and hasattr(args, 'use_cldice') and args.use_cldice:
        avg_cldice = np.mean(cl_dice_scores)
        std_cldice = np.std(cl_dice_scores)
        combined_score = 0.5 * avg_dice + 0.5 * avg_cldice
        
        if compute_loss and avg_val_loss is not None:
            print(f"  dice: {avg_dice:.4f}, cldice: {avg_cldice:.4f}, combined: {combined_score:.4f}, val_loss: {avg_val_loss:.4f}")
        else:
            print(f"  dice: {avg_dice:.4f}, cldice: {avg_cldice:.4f}, combined: {combined_score:.4f}")
        
        # Dice, clDice, Combined, Std, Val_Loss 순서로 반환
        return avg_dice, avg_cldice, combined_score, std_dice, std_cldice, avg_val_loss
    else:
        avg_cldice = None
        combined_score = None
        std_cldice = None
        
        if compute_loss and avg_val_loss is not None:
            print(f"  dice: {avg_dice:.4f}, val_loss: {avg_val_loss:.4f}")
        else:
            print(f"  dice: {avg_dice:.4f}")
        
        # clDice 미사용 시 (avg_cldice N/A, combined_score N/A, std_cldice N/A)
        return avg_dice, avg_cldice, combined_score, std_dice, std_cldice, avg_val_loss

def train_model(model, args, tr_loader, val_loader, epochs, param_path):
    """
    Train model with optimal validation loss calculation strategy.
    Validation loss is computed:
    - Epochs ~50% : Skipped (fast learning phase)
    - Epochs 50%~80% : Every 5 epochs (convergence check)
    - Epochs 80%~ : Every epoch (overfitting monitoring)
    """
    
    if args.use_fp16:
        scaler = GradScaler()
        iters_to_accumulate = args.use_fp16
    
    total = len(tr_loader.dataset)
    tr_iter = len(tr_loader)
    best_score = 0
    
    # validation loss 계산 기준 epoch 계산
    # ═══════════════════════════════════════
    epoch_50_percent = int(epochs * 0.5)
    epoch_80_percent = int(epochs * 0.8)
    
    # training_log
    # ═══════════════════════════════════════
    log_dir = args.train_log_path
    log_filename = f'{args.param_name}_training_log.csv'
    log_file = os.path.join(log_dir, log_filename)

    # 디렉토리가 없으면 생성
    os.makedirs(log_dir, exist_ok=True)
     
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Dice_Score,clDice_Score,Combined_Score,Dice_Std,clDice_std,Best_Score,Time\n")

    print(f"Training log will be saved to: {log_file}")

    for epoch in range(epochs):
        start_time = time.time()
        sum_loss_DICE = 0
        sum_loss_CL = 0
        sum_loss_total = 0
        
        model.train()
        
        for i, data in enumerate(tr_loader):
            _img, _mask = data['image'], data['mask']
            _img = _img.float().to(args.device)
            _img = _img.unsqueeze(1)
        
            _mask = _mask.float().to(args.device)
            
            w_dice = args.dice_w
            w_cl = args.cldice_w
            
            if args.use_fp16:
                with autocast(device_type="cuda"):
                    _out = model(_img)
                    
                    loss_DICE = args.criterion(_out, _mask)
                    
                    if args.use_cldice:
                        main_output = _out[0] if (args.model == 'U2net' and isinstance(_out, (tuple, list))) else _out
                        loss_CL = args.criterion_cl(main_output, _mask)
                        loss = (loss_DICE * w_dice + loss_CL * w_cl) / iters_to_accumulate
                    else:
                        loss = loss_DICE / iters_to_accumulate
                    
                scaler.scale(loss).backward()
                
                if (i+1) % iters_to_accumulate == 0:
                    scaler.unscale_(args.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(args.optimizer)
                    scaler.update()
                    args.optimizer.zero_grad()
                
                # Loss 누적
                sum_loss_DICE += loss_DICE.item()
                if args.use_cldice:
                    sum_loss_CL += loss_CL.item()
                    sum_loss_total += (loss_DICE.item() * w_dice + loss_CL.item() * w_cl) 
                else:
                    sum_loss_total += loss_DICE.item() 
                
            else:
                _out = model(_img) 

                loss_DICE = args.criterion(_out, _mask)

                # clDice 사용 시
                if args.use_cldice:
                    main_output = _out[0] if (args.model == 'U2net' and isinstance(_out, (tuple, list))) else _out
                    loss_CL = args.criterion_cl(main_output, _mask)
                    loss = (loss_DICE * w_dice + loss_CL * w_cl)
                    sum_loss_CL += loss_CL.item()
                else:
                    loss = loss_DICE

                loss.backward()

                sum_loss_DICE += loss_DICE.item()
                sum_loss_total += loss.item()

                args.optimizer.step()
                args.optimizer.zero_grad()
                
            if (i + 1) % 100 == 0:
                if hasattr(args, 'use_cldice') and args.use_cldice:
                    print(f"Epoch [{epoch}/{epochs-1}] Batch [{i+1}/{len(tr_loader)}] "
                          f"Dice: {loss_DICE.item():.4f} clDice: {loss_CL.item():.4f} "
                          f"Weights: ({w_dice:.1f},{w_cl:.1f}) Total: {loss.item():.4f}")
                else:
                    print(f"Epoch [{epoch}/{epochs-1}] Batch [{i+1}/{len(tr_loader)}] "
                          f"Seg: {loss_DICE.item():.4f}")


        # Validation loss 계산 여부 
        # ═══════════════════════════════════════
        if epoch < epoch_50_percent:
            compute_val_loss = False
        elif epoch < epoch_80_percent:
            compute_val_loss = (epoch % 10 == 0) 
        else:
            if epoch >= epochs - 10:
                compute_val_loss = True
            else:
                compute_val_loss = (epoch % 5 == 0)
        
        
        # Validation
        # ═══════════════════════════════════════
        dice_score, cldice_score, combined_score, dice_std, cldice_std, val_loss = validation(
            model, args, val_loader, compute_loss=compute_val_loss
        )
        
        
        # best_score 업데이트 및 모델 저장
        # ═══════════════════════════════════════
        current_score = combined_score if combined_score is not None else dice_score
        
        if best_score < current_score:
            best_score = current_score
            torch.save(model.module.state_dict(), param_path)
        
        # 평균 loss 계산
        running_loss_DICE = sum_loss_DICE/tr_iter
        running_loss_total = sum_loss_total/tr_iter
        time_elapsed = time.time() - start_time
        
        # Log 파일에 기록
        # ═══════════════════════════════════════
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
        cldice_score_str = f"{cldice_score:.6f}" if cldice_score is not None else "N/A"
        combined_score_str = f"{combined_score:.6f}" if combined_score is not None else "N/A"
        cldice_std_str = f"{cldice_std:.6f}" if cldice_std is not None else "N/A"
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{running_loss_total:.6f},{val_loss_str},"
                   f"{dice_score:.6f},{cldice_score_str},{combined_score_str},"
                   f"{dice_std:.6f},{cldice_std_str},{best_score:.6f},{time_elapsed:.2f}\n")
        
        # 출력
        val_loss_display = f"%.6f" % val_loss if val_loss is not None else "N/A (skipped)"
        
        if hasattr(args, 'use_cldice') and args.use_cldice:
            running_loss_CL = sum_loss_CL/tr_iter
            print('Epoch {}/{}'.format(epoch, epochs -1),
                  'loss_DICE: %.6f' %(running_loss_DICE),
                  'loss_CL: %.6f' %(running_loss_CL),
                  'loss_total: %.6f' %(running_loss_total),
                  '| val_loss:', val_loss_display,
                  '| dice: %.4f' %(dice_score),
                  '| cldice: %.4f' %(cldice_score),
                  '| combined: %.4f' %(combined_score),
                  '| best: %.4f' %(best_score), 
                  '| dice_std: %.4f' %(dice_std),
                  '| clDice_std: %.4f' %(cldice_std),
                  '| time: %.2f'%(time_elapsed))
        else:
            print('Epoch {}/{}'.format(epoch, epochs -1),
                  'loss_DICE: %.6f' %(running_loss_DICE),
                  '| val_loss:', val_loss_display,
                  '| dice: %.4f' %(dice_score),
                  '| best: %.4f' %(best_score), 
                  '| dice_std: %.4f' %(dice_std),
                  '| time: %.2f'%(time_elapsed))
    
    # 학습 완료 후 최종 그래프 생성
    # ═══════════════════════════════════════
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation score: {best_score:.6f}")
    print(f"Model saved: {param_path}")
    print("="*60)
    
    # 로그 파일에서 그래프 생성
    curve_filename = f'{args.param_name}_training_curves.png'
    curve_path = os.path.join(log_dir, curve_filename)
    plot_training_curves_from_log(log_file, curve_path)

def main():
    print("Clean code!")
    args = parse_args()
    
    if args.model == 'U2net':
        # 출력 채널을 1로 변경
        model = U2net.U2NET(1, 1)
        
        # clDice 사용 여부에 따라 손실 함수 설정
        if args.use_cldice:
            # Supv_DiceLoss는 U2net의 deep supervision용
            args.criterion = U.Supv_DiceLoss()
            # clDice는 전경에만 적용 (exclude_background=False, 단일 채널이므로 불필요)
            args.criterion_cl = soft_dice_cldice(
                iter_=args.skel_iter,  # 파라미터 적용
                alpha=args.alpha, 
                smooth=1., 
                exclude_background=False
             )
        else:
            args.criterion = U.Supv_DiceLoss()
        
    elif args.model == 'UnetPP':
        model = smp.UnetPlusPlus('efficientnet-b4', encoder_weights='imagenet',
                         activation = 'sigmoid',
                         in_channels=1, classes=1)
        
        if args.use_cldice:
            args.criterion = U.DiceLoss()
            args.criterion_cl = soft_cldice(
                iter_=3,
                alpha=args.alpha,
                smooth=1.,
                exclude_background=False
            )
        else:
            args.criterion = U.DiceLoss()
        
    elif args.model == 'Deeplab':
        model = smp.DeepLabV3Plus('efficientnet-b4', encoder_weights='imagenet',
                         activation = 'sigmoid',
                         in_channels=1, classes=1)
        
        if args.use_cldice:
            args.criterion = U.DiceLoss()
            args.criterion_cl = soft_cldice(
                iter_=3,
                alpha=args.alpha,
                smooth=1.,
                exclude_background=False
            )
        else:
            args.criterion = U.DiceLoss()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        import sys
        sys.exit("CUDA is not available")
        
    model = nn.DataParallel(model).to(args.device)

    args.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
    augment = select_aug(['Translation', 'GaussNoise'])
    
    if args.dataset == 'CBN':                     
        data_path = "/CBN/yi/QCA/10. DatasetForDL/CBN/meta.json"
        with open(data_path) as json_file:
            meta_file = json.load(json_file)
    
    tr_set, val_set, te_set = U.select_fold(args.fold, meta_file["fold_idx"])
    
    tr_set = QCAdataset(tr_set, meta_file['meta'], augment=augment)
    val_set = QCAdataset(val_set, meta_file['meta'])
    te_set = QCAdataset(te_set, meta_file['meta'])
    
    train_loader = DataLoader(tr_set, batch_size = args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size = args.batch_size,
                            shuffle=False, num_workers=4)
    te_loader = DataLoader(te_set, batch_size= args.batch_size,
                       shuffle=False, num_workers=4)
    
    param_path = args.param_path + args.param_name
    
    # Train
    train_model(model, args, train_loader, val_loader,
                epochs=args.epochs, param_path=param_path)

    # test는 test_model.py에서 별도로 진행함.
    
if __name__ == '__main__':
    main()