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

from clDice.cldice_loss.pytorch.cldice import soft_dice_cldice

def to_float32(x):
    """Convert numpy array to float32 dtype"""
    return x.astype(np.float32)

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


def load_model(model_name, model_path, device):

    # 모델 아키텍처 초기화
    if model_name == 'U2net':
        model = U2net.U2NET(1, 1)
        
    elif model_name == 'UnetPP':
        model = smp.UnetPlusPlus('efficientnet-b4', encoder_weights='imagenet',
                            activation='sigmoid',
                            in_channels=1, classes=1)
        
    elif model_name == 'Deeplab':
        model = smp.DeepLabV3Plus('efficientnet-b4', encoder_weights='imagenet',
                            activation='sigmoid',
                            in_channels=1, classes=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # 모델 파라미터 로드
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = nn.DataParallel(model).to(device)
    
    return model


def test_model(model, args, dataset_name, data_path):

    with open(data_path) as json_file:
        meta_file = json.load(json_file)
    
    # 모든 fold의 test set 통합
    te_set = []
    for i in range(5):
        te_set += meta_file["fold_idx"][i]
    
    te_set = QCAdataset(te_set, meta_file['meta'], dataset=dataset_name)
    te_loader = DataLoader(te_set, batch_size=args.batch_size,
                          shuffle=False, num_workers=4)
    
    # validation 계산에서는 comput_loss = False!
    dice_avg, cldice_avg, combined_score, dice_std, cldice_std = validation(
        model, args, te_loader
    )
    
    if hasattr(args, 'use_cldice') and args.use_cldice:
         # 결과 출력
        print(f'[{dataset_name} results] ===========================================')
        print('combined_score: %.6f' % (combined_score))
        print('dice_avg: %.6f' % (dice_avg))
        print('cldice_avg: %.6f' % (cldice_avg))
        print('dice_std: %.6f' % (dice_std))
        print('cldice_std: %.6f' % (cldice_std))
        print('=====================================================\n')
    else:
        print(f'[{dataset_name} results] ===========================================')
        print('dice_avg: %.6f' % (dice_avg))
        print('dice_std: %.6f' % (dice_std))
        print('=====================================================\n')
    
    return {
        'combined_score': combined_score,
        'dice_avg': dice_avg,
        'cldice_avg': cldice_avg,
        'dice_std': dice_std,
        'cldice_std': cldice_std
        
    }

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
        
        # U2Net의 경우 main output 추출
        if args.model_name == 'U2net':
            pred_seg = pred_seg[0]
        
        
        # GPU 마스크 준비 (loss 계산 또는 clDice 사용 시 필요)
        # ═══════════════════════════════════════
        need_mask_gpu = (hasattr(args, 'use_cldice') and args.use_cldice)
        if need_mask_gpu:
            _mask_gpu = _mask.float().to(args.device)
                    
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
    
    # Combined metric 계산
    if cl_dice_scores and hasattr(args, 'use_cldice') and args.use_cldice:
        avg_cldice = np.mean(cl_dice_scores)
        std_cldice = np.std(cl_dice_scores)
        combined_score = 0.5 * avg_dice + 0.5 * avg_cldice
        
        print(f"  dice: {avg_dice:.4f}, cldice: {avg_cldice:.4f}, combined: {combined_score:.4f}")
        
        # Dice, clDice, Combined, Std, Val_Loss 순서로 반환
        return avg_dice, avg_cldice, combined_score, std_dice, std_cldice
    else:
        avg_cldice = None
        combined_score = None
        std_cldice = None
        
        print(f"  dice: {avg_dice:.4f}")
        
        # clDice 미사용 시 (avg_cldice N/A, combined_score N/A, std_cldice N/A)
        return avg_dice, avg_cldice, combined_score, std_dice, std_cldice



def main():
    parser = argparse.ArgumentParser(description='Test vessel segmentation models')
    
    # use_cldice
    parser.add_argument('--use_cldice', action='store_true',
                       help='Use clDice loss for topology preservation')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for clDice loss (0.0=only dice, 1.0=only cldice)')
    parser.add_argument('--skel_iter', type=int, default=10,
                       help='Number of iterations for soft skeletonization')
    
    # 모델 관련 arguments
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['U2net', 'UnetPP', 'Deeplab'],
                       help='Model architecture name')
    parser.add_argument('--model_file', type=str, required=True,
                       help='Model filename (e.g., best_model.pth)')
    parser.add_argument('--param_path', type=str, default='./param/',
                       help='The model parameter will be saved at this path.')
    
    # 테스트 관련 arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for testing')
    
    # 데이터셋 경로
    parser.add_argument('--cbn_data_path', type=str,
                       default='/CBN/yi/QCA/10. DatasetForDL/CBN/meta.json',
                       help='Path to CBN meta.json')
    parser.add_argument('--cuh_data_path', type=str,
                       default='/CBN/yi/QCA/10. DatasetForDL/CUH/meta.json',
                       help='Path to CUH meta.json')
    
    args = parser.parse_args()
    
    #일단 model이 U2net일 경우에만!
    if args.model_name == 'U2net':
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
            
            
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        import sys
        sys.exit("CUDA is not available")
    
    
    # 모델 경로
    param_path = args.param_path + args.model_file
    
    print(f'\n{"="*60}')
    print(f'Testing Model: {args.model_name}')
    print(f'Model Path: {param_path}')
    print(f'Device: {args.device}')
    print(f'{"="*60}\n')
    
    # 모델 로드
    model = load_model(args.model_name, param_path, args.device)
    
    # CBN 데이터셋 테스트
    print("Testing on CBN dataset...")
    cbn_results = test_model(model, args, 'CBN', args.cbn_data_path)
    
    # CUH 데이터셋 테스트
    print("Testing on CUH dataset...")
    cuh_results = test_model(model, args, 'CUH', args.cuh_data_path)
    
    # 전체 결과 요약
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    if hasattr(args, 'use_cldice') and args.use_cldice:
        print(f'CBN - Dice: {cbn_results["dice_avg"]:.6f} ± {cbn_results["dice_std"]:.6f}, clDice: {cbn_results["cldice_avg"]:.6f} ± {cbn_results["cldice_std"]:.6f}, Combined: {cbn_results["combined_score"]:.6f}')
        print(f'CUH - Dice: {cuh_results["dice_avg"]:.6f} ± {cuh_results["dice_std"]:.6f}, clDice: {cuh_results["cldice_avg"]:.6f} ± {cuh_results["cldice_std"]:.6f}, Combined: {cuh_results["combined_score"]:.6f}')
    else:
        print(f'CBN - Dice: {cbn_results["dice_avg"]:.6f} ± {cbn_results["dice_std"]:.6f}')
        print(f'CUH - Dice: {cuh_results["dice_avg"]:.6f} ± {cuh_results["dice_std"]:.6f}')
        
    print(f'{"="*60}\n')

if __name__ == '__main__':
    main()