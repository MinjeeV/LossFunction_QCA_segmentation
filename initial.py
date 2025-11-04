import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.v2 import GaussianNoise

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler
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

from clDice.cldice_loss.pytorch.cldice import soft_dice_cldice, soft_cldice 


def to_float32(x):
    """Convert numpy array to float32 dtype"""
    return x.astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--param_name', type=str, default='TEST00_01',
                       help='The model parameter will be saved as this name.')
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
    
    # clDice 관련 파라미터
    parser.add_argument('--use_cldice', action='store_true',
                       help='Use clDice loss for topology preservation')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for clDice loss (0.0=only dice, 1.0=only cldice)')
    parser.add_argument('--skel_iter', type=int, default=10,
                       help='Number of iterations for soft skeletonization')
    
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

def validation(model, args, data_loader, thres_prop=False):
    dice_scores = []
    cl_dice_scores = []
    model.eval()
    
    for i, data in enumerate(data_loader):
        _img, _mask = data['image'], data['mask']
        _img = _img.float().to(args.device)
        _img = _img.unsqueeze(1)

        _mask_np = _mask.cpu().detach().numpy()

        with torch.no_grad():
            _out = model(_img)
            pred_seg = _out
                    
        if args.model == 'U2net':
            pred_seg = pred_seg[0]
        
        # 디버깅: 첫 배치에서 shape 확인
        if i == 0:
            print(f"[Validation Debug]")
            print(f"  pred_seg before threshold - shape: {pred_seg.shape}, min: {pred_seg.min():.4f}, max: {pred_seg.max():.4f}")
            print(f"  _mask shape: {_mask_np.shape}, unique values: {np.unique(_mask_np)}")
        
        # Dice score 계산 (기존)
        pred_binary = (pred_seg>0.5).type(torch.uint8).cpu().detach().numpy()
        
        if i == 0:
            print(f"  pred_binary shape: {pred_binary.shape}, unique values: {np.unique(pred_binary)}")
            
        if thres_prop != False:
            pred_binary = U.post_processing(pred_binary, thres_prop=thres_prop)
            
        # 단일 채널 Dice score 계산
        Score = U.calculate_score(_mask_np[:,0], pred_binary[:,0], 'Dice')
        dice_scores.extend(Score)
        
        # clDice score 계산 (topology 평가)
        if hasattr(args, 'use_cldice') and args.use_cldice:
            _mask_tensor = _mask.float().to(args.device)
            with torch.no_grad():
                # clDice loss는 낮을수록 좋으므로, 1에서 빼서 score로 변환
                cl_loss = args.criterion_cl(pred_seg, _mask_tensor).item()
                cl_score = 1.0 - cl_loss
                cl_dice_scores.append(cl_score)
        
    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    # Combined metric 계산
    if cl_dice_scores and hasattr(args, 'use_cldice') and args.use_cldice:
        avg_cldice = np.mean(cl_dice_scores)
        # 0.5 * Dice + 0.5 * clDice
        combined_score = 0.5 * avg_dice + 0.5 * avg_cldice
        
        print(f"  avg_dice: {avg_dice:.4f}, avg_cldice: {avg_cldice:.4f}, combined: {combined_score:.4f}")
        
        return combined_score, std_dice, avg_dice, avg_cldice
    else:
        return avg_dice, std_dice, avg_dice, 0.0

def get_loss_weights(epoch, total_epochs):
    """동적 loss weight 계산"""
    if epoch <= 50:
        # Epoch 0~50: (0.7, 0.3) 고정
        alpha = 0.7
        beta = 0.3
    else:
        # Epoch 51~90: 0.7->0.1, 0.3->0.9 (40 epoch에 걸쳐 변화)
        progress = min((epoch - 50) / 40, 1.0)
        alpha = 0.7 - 0.6 * progress  # 0.7 -> 0.1
        beta = 0.3 + 0.6 * progress   # 0.3 -> 0.9
    
    return alpha, beta


def train_model(model, args, tr_loader, val_loader, epochs, param_path):
    
    if args.use_fp16:
        scaler = GradScaler()
        iters_to_accumulate = args.use_fp16
    
    total = len(tr_loader.dataset)
    tr_iter = len(tr_loader)
    best_score = 0

    for epoch in range(epochs):
        start_time = time.time()
        sum_loss_DICE = 0
        sum_loss_CL = 0
        sum_loss_total = 0
        
        # 동적 loss weight 계산
        if args.use_cldice:
            alpha, beta = get_loss_weights(epoch, epochs)
        
        model.train()
        
        for i, data in enumerate(tr_loader):
            _img, _mask = data['image'], data['mask']
            _img = _img.float().to(args.device)
            _img = _img.unsqueeze(1)
        
            _mask = _mask.float().to(args.device)
            
            if args.use_fp16:
                with amp.autocast("cuda"):
                    _out = model(_img)
                    
                    loss_DICE = args.criterion(_out, _mask)
                    
                    if args.use_cldice:
                        # U2net의 경우 main output만 사용
                        main_output = _out[0] if (args.model == 'U2net' and isinstance(_out, (tuple, list))) else _out
                        loss_CL = args.criterion_cl(main_output, _mask)
                        loss = (loss_DICE * alpha + loss_CL * beta) / iters_to_accumulate
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
                    sum_loss_total += (loss_DICE.item() * alpha + loss_CL.item() * beta) 
                else:
                    sum_loss_total += loss_DICE.item() 
                
            else:
                _out = model(_img) 

                loss_DICE = args.criterion(_out, _mask)

                # clDice 사용 시
                if args.use_cldice:
                    
                    main_output = _out[0] if (args.model == 'U2net' and isinstance(_out, (tuple, list))) else _out
                    loss_CL = args.criterion_cl(main_output, _mask)
                    loss = (loss_DICE * alpha + loss_CL * beta)
                    sum_loss_CL += loss_CL.item()
                    
                else:
                    loss = loss_DICE

                loss.backward()

                sum_loss_DICE += loss_DICE.item()
                sum_loss_total += loss.item()

                args.optimizer.step()
                args.optimizer.zero_grad()
                
            if (i + 1) % 50 == 0:
                if hasattr(args, 'use_cldice') and args.use_cldice:
                    print(f"Epoch [{epoch}/{epochs-1}] Batch [{i+1}/{len(tr_loader)}] "
                          f"Dice: {loss_DICE.item():.4f} clDice: {loss_CL.item():.4f} "
                          f"Total: {loss.item():.4f} (α={alpha:.2f}, β={beta:.2f})")
                else:
                    print(f"Epoch [{epoch}/{epochs-1}] Batch [{i+1}/{len(tr_loader)}] "
                          f"Seg: {loss_DICE.item():.4f} (α={alpha:.2f}, β={beta:.2f})")

        total_score, dice_std, dice_avg, cldice_avg = validation(model, args, val_loader)
        
        if best_score < total_score:
            best_score = total_score
            torch.save(model.module.state_dict(), param_path)
        
        running_loss_DICE = sum_loss_DICE/tr_iter
        running_loss_total = sum_loss_total/tr_iter

        time_elapsed = time.time() - start_time
        
        if hasattr(args, 'use_cldice') and args.use_cldice:
            running_loss_CL = sum_loss_CL/tr_iter
            print('Epoch {}/{}'.format(epoch, epochs -1),
                  'avg_loss_DICE: %.6f' %(running_loss_DICE),
                  'avg_loss_CL: %.6f' %(running_loss_CL),
                  'avg_loss_total: %.6f' %(running_loss_total),
                  'α: %.2f' %(alpha), 'β: %.2f' %(beta),
                  '| best_score: %.6f' %(best_score), 
                  '| dice_std: %.6f' %(dice_std),
                  '| time: %.2f'%(time_elapsed))
        else:
            print('Epoch {}/{}'.format(epoch, epochs -1),
                  'avg_loss_DICE: %.6f' %(running_loss_DICE),
                  '| best_score: %.6f' %(best_score), 
                  '| dice_std: %.6f' %(dice_std),
                  '| time: %.2f'%(time_elapsed))


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
                iter_=3, 
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

    # Testing
    if args.model == 'U2net':
        model = U2net.U2NET(1, 1)
        
    elif args.model == 'UnetPP':
        model = smp.UnetPlusPlus('efficientnet-b4', encoder_weights='imagenet',
                         activation = 'sigmoid',
                         in_channels=1, classes=1)
        
    elif args.model == 'Deeplab':
        model = smp.DeepLabV3Plus('efficientnet-b4', encoder_weights='imagenet',
                         activation = 'sigmoid',
                         in_channels=1, classes=1)
    
    model.load_state_dict(torch.load(param_path, weights_only=True))
    model = nn.DataParallel(model).to(args.device)
    total_score, dice_std, dice_avg, cldice_avg = validation(model, args, te_loader)
    
    print('[CBN results] ===========================================')
    print('total_score: %.6f' %(total_score))
    print('dice_std: %.6f' %(dice_std))
    print('dice_avg: %.6f' %(dice_avg))
    print('cldice_avg: %.6f' %(cldice_avg))
    print('=====================================================')

    
    # CUH Results    
    data_path = "/CBN/yi/QCA/10. DatasetForDL/CUH/meta.json"
    with open(data_path) as json_file:
        meta_file = json.load(json_file)
    
    te_set = []
    for i in range(5):
        te_set += meta_file["fold_idx"][i]

    te_set = QCAdataset(te_set, meta_file['meta'], dataset='CUH')     
    te_loader = DataLoader(te_set, batch_size= args.batch_size,
                       shuffle=False, num_workers=4)
    
    total_score, dice_std, dice_avg, cldice_avg = validation(model, args, te_loader)

    print('[CUH results] ===========================================')
    print('total_score: %.6f' %(total_score))
    print('dice_std: %.6f' %(dice_std))
    print('dice_avg: %.6f' %(dice_avg))
    print('cldice_avg: %.6f' %(cldice_avg))
    print('=====================================================')
    
if __name__ == '__main__':
    main()