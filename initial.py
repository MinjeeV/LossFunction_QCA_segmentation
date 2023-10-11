import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import segmentation_models_pytorch as smp
import albumentations as album
import U2net

import wandb
import utils as U

import cv2
import numpy as np
import time, os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('param_name', type=str, default='TEST00_00',
                       help='The model parameter will be saved as this name.')
    parser.add_argument('--param_path', type=str, default='./param/',
                       help='The model parameter will be saved at this path.')
    parser.add_argument('--project', type=str, default='QCA_segmentation',
                       help='The model parameter will be saved at this path.')
    
    parser.add_argument('--model', choices=['U2net', 'Deeplab', 'UnetPP'],
                        default='U2net')
    parser.add_argument('--dataset', type=str, default='CBN')
    parser.add_argument('--fold', type=int, default= 0)
    
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=400)
    
    args = parser.parse_args()
    return args

def select_aug(aug_list):
    transforms = []
    for i in aug_list:
        if i == 'Translation':
            transforms.append(
                album.ShiftScaleRotate(scale_limit=(-0.2,0),
                                      rotate_limit=20,
                                      shift_limit=0.1,
                                      border_mode=0, value=[0.3,0.4,0.5],
                                      p=1))
        elif i == 'Contrast':
            transforms.append(album.RandomContrast(limit=0.4, p=0.5))
        
        elif i == 'GaussNoise':
            transforms.append(album.GaussNoise(var_limit=(0, 0.01), p=0.5))
        elif i == 'GaussBlur':
            transforms.append(album.GaussianBlur(blur_limit=5, p=0.2))
        elif i == 'Gamma':
            transforms.append(album.RandomGamma(gamma_limit=(50,150), p=0.2))
        else:
            print('name error')
            return
            
    return album.Compose(transforms = transforms)

class QCAdataset(Dataset):
    def __init__(self, idxs, meta, augment=False, dataset='CBN'):
        self.idxs = idxs # tr, val, te idxs
        self.meta = meta
        self.base_path = '/CBN/Angiography/1. QCA/10. DatasetForDL/'+dataset
        self.augment = augment
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        _path = os.path.join(self.base_path, 
                                'Input', 'png', 
                             self.meta[self.idxs[idx]]['id']+'.png') # fixed
        img = cv2.imread(_path)[:,:,0]
        
        _path = os.path.join(self.base_path, 
                                'Mask', 'extracted', 
                             self.meta[self.idxs[idx]]['id']+'.png')
        mask = cv2.imread(_path)    
            
        mm = np.min(img)
        sample = {'image':(img-mm)/(np.max(img)-mm),
                  'mask':mask/255}

        if self.augment:
            sample = self.augment(**sample)
        
        sample ={'image': sample['image'],
                 'mask': np.transpose(sample['mask'], (2,0,1)),
                 'class': self.meta[self.idxs[idx]]['class']}
        return sample

def train_model(model, args, tr_loader, val_loader, epochs, param_path):
    run = wandb.init(project=args.project, name=args.param_name+'_'+args.model,
                     job_type=args.dataset+'_train', config=args.config)
    
    total = len(tr_loader.dataset)
    print('total # of train data:', total)
    print(args.config)
    tr_iter = len(tr_loader)
    best_score = 0

    for epoch in range(epochs):
        start_time = time.time()
        sum_loss_DICE = 0
        sum_loss_CE = 0
        model.train()
        for i, data in enumerate(tr_loader):
            # set model input
            _img, _mask, _cl = data['image'], data['mask'], data['class']
            _img = _img.float().to(args.device)
            _img = _img.unsqueeze(1)
        
            _mask = _mask.float().to(args.device)
            _cl = _cl.long().to(args.device)
            
            _out = model(_img)
            loss_DICE = args.criterion(_out[0], _mask)
            loss_CE = args.criterion_cl(_out[1], _cl)
            sum_loss_DICE += loss_DICE.item()
            sum_loss_CE += loss_CE.item()

            loss = (loss_DICE + 0.1*loss_CE)
            loss.backward()
            args.optimizer.step()
            args.optimizer.zero_grad()
            
        score, f1, _, _acc = validation(model, args, val_loader)
        
        if best_score < score:
            best_score = score
            torch.save(model.module.state_dict(), param_path)
        
        running_loss_DICE = sum_loss_DICE/tr_iter
        running_loss_CE = sum_loss_CE/tr_iter

        time_elapsed = time.time() - start_time
        
        # wandb log save
        wandb.log({'train_loss': running_loss_DICE,
                   'val_avg_f1': score,
                   'val_acc': _acc,
                   'val_best_score': best_score,
                   'lr': args.optimizer.param_groups[0]['lr']})
                
        
        print('Epoch {}/{}'.format(epoch, epochs -1),
              'avg_loss_DICE: %.6f' %(running_loss_DICE),
              'avg_loss_CE: %.6f' %(running_loss_CE),
              'accuracy: %.6f' %(_acc),
              '| best_score: %.6f' %(best_score), 
              '| time: %.2f'%(time_elapsed))
    run.finish()
        
def validation(model, args, data_loader, thres_prop=False):
    score_M1 = []
    score_M2 = []
    score_M3 = []
    running_corrects = 0.0
    model.eval()
    for i, data in enumerate(data_loader):
        # set model input
        _img, _mask, _cl = data['image'], data['mask'], data['class']
        _img = _img.float().to(args.device)
        _img = _img.unsqueeze(1)

        _cl = _cl.long().to(args.device)
        
        _mask = _mask.cpu().detach().numpy()

        with torch.no_grad():
            _out = model(_img)
            _, pred_cl = torch.max(_out[1], 1)
            pred_seg = _out[0]
            
        running_corrects += pred_cl.eq(_cl).sum().item()
                    
        if args.model == 'U2net':
            pred_seg = pred_seg[0]
        
        pred_seg = (pred_seg>0.5).type(torch.uint8)
        pred_seg = pred_seg.cpu().detach().numpy()
            
        if thres_prop != False:
            _out = U.post_processing(_out, thres_prop=thres_prop)
            
        Score1 = U.calculate_score(_mask[:,0], pred_seg[:,0],'Dice')
        Score2 = U.calculate_score(_mask[:,1], pred_seg[:,1],'Dice')
        Score3 = U.calculate_score(_mask[:,2], pred_seg[:,2],'Dice')
        
        score_M1.extend(Score1)
        score_M2.extend(Score2)
        score_M3.extend(Score3)
        
    avg_score_M1 = np.mean(score_M1)
    avg_score_M2 = np.mean(score_M2)
    avg_score_M3 = np.mean(score_M3)
    
    avg_score = np.mean([avg_score_M1, avg_score_M2, avg_score_M3])
    
    epoch_acc = running_corrects/len(data_loader.dataset)
        
    std_M1 = np.std(score_M1)
    std_M2 = np.std(score_M2)
    std_M3 = np.std(score_M3)
    
    f1 = (avg_score_M1, avg_score_M2, avg_score_M3)
    std = (std_M1, std_M2, std_M3)
    return avg_score, f1, std, epoch_acc

def make_config(args):
    cfg = {
        'experiment': args.param_name,
        'dataset': args.dataset,
        'model': args.model,
        'input_size': 512,
        'batch_size': args.batch_size}
    return cfg

def main():
    args = parse_args()
    args.config = make_config(args) # for wandb
    wandb.login()
    
    aux_params = dict(pooling='avg',
                      dropout=0.5,
                      classes=3)
    if args.model == 'U2net':
        model = U2net.U2NET_CL(1,3, aux_params=aux_params)
        args.criterion = U.Supv_DiceLoss()
    elif args.model == 'UnetPP':
        model = smp.UnetPlusPlus('efficientnet-b4', encoder_weights='imagenet',
                         activation = 'sigmoid', aux_params=aux_params,
                         in_channels=1, classes=3)
        args.criterion = U.DiceLoss()
    elif args.model == 'Deeplab':
        model = smp.DeepLabV3Plus('efficientnet-b4', encoder_weights='imagenet',
                         activation = 'sigmoid', aux_params=aux_params,
                         in_channels=1, classes=3)
        args.criterion = U.DiceLoss()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(model).to(args.device)
    print(args.device)
                             
    args.criterion_cl = nn.CrossEntropyLoss()
    args.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    augment = select_aug(['Contrast', 'Translation', 'GaussNoise'])
    
    if args.dataset == 'CBN':                     
        data_path = "/CBN/Angiography/1. QCA/10. DatasetForDL/CBN/meta.json"
        with open(data_path) as json_file:
            meta_file = json.load(json_file)
    
    tr_set, val_set, te_set = U.select_fold(args.fold, meta_file["fold_idx"])
    tr_set = QCAdataset(tr_set, meta_file['meta'], augment=augment)
    val_set = QCAdataset(val_set, meta_file['meta'])
    te_set = QCAdataset(te_set, meta_file['meta'])
    
    train_loader = DataLoader(tr_set, batch_size = args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size = args.batch_size*2,
                            shuffle=False, num_workers=4)
    te_loader = DataLoader(te_set, batch_size= args.batch_size*2,
                       shuffle=False, num_workers=4)
    
    param_path = args.param_path + args.param_name
    # Train
    train_model(model, args, train_loader, val_loader,
                epochs=args.epochs, param_path=param_path)
    
    # Test
    if args.model == 'U2net':
        model = U2net.U2NET_CL(1,3, aux_params=aux_params)
    elif args.model == 'UnetPP':
        model = smp.UnetPlusPlus('efficientnet-b4', encoder_weights='imagenet',
                         activation = 'sigmoid', aux_params=aux_params,
                         in_channels=1, classes=3)
    elif args.model == 'Deeplab':
        model = smp.DeepLabV3Plus('efficientnet-b4', encoder_weights='imagenet',
                         activation = 'sigmoid', aux_params=aux_params,
                         in_channels=1, classes=3)
    
    model.load_state_dict(torch.load(param_path))
    model = nn.DataParallel(model).to(args.device)
    avg_score, f1, std, _acc = validation(model, args, te_loader)
    
    # Results
    run = wandb.init(project=args.project, name=args.param_name+'_'+args.model,
                     job_type='CBN_test', config=args.config)
    results = wandb.Table(columns = ['dice_m1', 'dice_m2', 'dice_m3', 'avg', 'acc'])
    results.add_data(f1[0], f1[1], f1[2], avg_score, _acc)
    wandb.log({'Test results': results})
    run.finish()
    
    print('[CBN results] ===========================================')
    print('classification accuracy:', _acc)
    print('avg_score: %.6f' %(avg_score))
    print('score_m1: %.6f' %(f1[0]),'std_m1: %.6f' %(std[0]))
    print('score_m2: %.6f' %(f1[1]),'std_m2: %.6f' %(std[1]))
    print('score_m3: %.6f' %(f1[2]),'std_m3: %.6f' %(std[2]))
    print('=====================================================')
    
    # CUH Results                               
    data_path = "/CBN/Angiography/1. QCA/10. DatasetForDL/CUH/meta.json"
    with open(data_path) as json_file:
        meta_file = json.load(json_file)
    # 모든 fold 합쳐서 prediction
    te_set = []
    for i in range(5):
        te_set += meta_file["fold_idx"][i]

    te_set = QCAdataset(te_set, meta_file['meta'], dataset='CUH')     
    te_loader = DataLoader(te_set, batch_size= args.batch_size*2,
                       shuffle=False, num_workers=4)
    
    avg_score, f1, std, _acc = validation(model, args, te_loader)
    run = wandb.init(project=args.project, name=args.param_name+'_'+args.model,
                     job_type='CUH_test', config=args.config)
    results = wandb.Table(columns = ['dice_m1', 'dice_m2', 'dice_m3', 'avg', 'acc'])
    results.add_data(f1[0], f1[1], f1[2], avg_score, _acc)
    wandb.log({'Test results': results})
    run.finish()
    print('[CUH results] ===========================================')
    print('classification accuracy:', _acc)
    print('avg_score: %.6f' %(avg_score))
    print('score_m1: %.6f' %(f1[0]),'std_m1: %.6f' %(std[0]))
    print('score_m2: %.6f' %(f1[1]),'std_m2: %.6f' %(std[1]))
    print('score_m3: %.6f' %(f1[2]),'std_m3: %.6f' %(std[2]))
    print('=====================================================')
    
        
if __name__ == '__main__':
    main()
    
    
    