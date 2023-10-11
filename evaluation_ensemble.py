import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import segmentation_models_pytorch as smp
import albumentations as album
import U2net

import sklearn.metrics as metrics

import wandb
import utils as U

import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2
import numpy as np
import time, os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str, default='TEST00_00',
                       help='Experiment will be saved as this name')
    parser.add_argument('--param_path', type=str, default='./param/',
                       help='The model parameter will be saved at this dir.')
    parser.add_argument('--project', type=str, default='QCA_segmentation',
                       help='The model parameter will be saved at this path.')
    
    parser.add_argument('--dataset', type=str, default='CBN')
    parser.add_argument('--fold', type=int, default= 0)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--vis_path', type=str, default='')
    
    args = parser.parse_args()
    return args

class QCAdataset(Dataset):
    def __init__(self, idxs, meta, augment=False, dataset='CBN'):
        self.idxs = idxs
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
                             self.meta[self.idxs[idx]]['id']+'.png')
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
        
def inference(model, args, data_loader, u2net=False,
               with_label=False, thres_prop=False):
    prediction = {'mask':[], 'class':[]}
    if with_label:
        label = {'img':[], 'mask':[], 'class':[]}
    else:
        label = False
        
    model.eval()
    for i, data in enumerate(data_loader):
        # set model input
        _img, _mask, _cl = data['image'], data['mask'], data['class']
        if with_label:
            label['img'].extend((_img*255).type(torch.uint8).detach().numpy())
            label['mask'].extend(_mask.type(torch.uint8).detach().numpy())
            label['class'].extend(_cl.type(torch.uint8).detach().numpy())
            
        _img = _img.float().to(args.device)
        _img = _img.unsqueeze(1)

        with torch.no_grad():
            _out = model(_img)
            _, pred_cl = torch.max(_out[1], 1)
            pred_seg = _out[0]
        
        if u2net:
            pred_seg = pred_seg[0]
        
        pred_seg = (pred_seg>0.5).type(torch.uint8)
        pred_seg = pred_seg.cpu().detach().numpy()
            
        if thres_prop != False:
            _out = U.post_processing(_out, thres_prop=thres_prop)
            
        prediction['mask'].extend(pred_seg)
        prediction['class'].extend(pred_cl.cpu().detach().numpy().astype(np.uint8))
    
    def list2arr(_dict):
        for k in _dict.keys():
            _dict[k] = np.array(_dict[k])
    
    list2arr(prediction)
    if with_label:
        list2arr(label)
    return prediction, label

def make_config(args):
    cfg = {
        'experiment': args.exp_name,
        'dataset': args.dataset,
        'model': 'Ensemble',
        'input_size': 512}
    return cfg

def main():
    args = parse_args()
    args.config = make_config(args) # for wandb
    wandb.login()
    
    # Model setting
    aux_params = dict(pooling='avg',
                      dropout=0.5,
                      classes=3)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                               
    model1 = U2net.U2NET_CL(1,3, aux_params=aux_params)
    model1.load_state_dict(torch.load(args.param_path+'A00_00'))
    model1 = nn.DataParallel(model1).to(args.device)
    
    model2 = smp.UnetPlusPlus('efficientnet-b4',
                     activation = 'sigmoid', aux_params=aux_params,
                     in_channels=1, classes=3)
    model2.load_state_dict(torch.load(args.param_path+'A00_01'))
    model2 = nn.DataParallel(model2).to(args.device)
    
    model3 = smp.UnetPlusPlus('efficientnet-b4',
                     activation = 'sigmoid', aux_params=aux_params,
                     in_channels=1, classes=3)
    model3.load_state_dict(torch.load(args.param_path+'A00_01'))
    model3 = nn.DataParallel(model3).to(args.device)
    
    
    # Data setting
    if args.dataset == 'CBN':
        data_path = "/CBN/Angiography/1. QCA/10. DatasetForDL/CBN/meta.json"
        with open(data_path) as json_file:
            meta_file = json.load(json_file)
        _, _, te_set = U.select_fold(args.fold, meta_file["fold_idx"])
        te_set = QCAdataset(te_set, meta_file['meta'])
    elif args.dataset == 'CUH':
        data_path = "/CBN/Angiography/1. QCA/10. DatasetForDL/CUH/meta.json"
        with open(data_path) as json_file:
            meta_file = json.load(json_file)
        # 모든 fold 합쳐서 prediction
        te_set = []
        for i in range(5):
            te_set += meta_file["fold_idx"][i]
        te_set = QCAdataset(te_set, meta_file['meta'], dataset='CUH')
            
    te_loader = DataLoader(te_set, batch_size= args.batch_size,
                       shuffle=False, num_workers=4)

    # Prediction
    models_pred = []
    for i, model in enumerate([model1, model2, model3]):
        if i == 0: #u2net
            u2net = True
            prediction, label = inference(model, args, te_loader, # Label data 
                                          u2net, with_label=True)
        else:
            u2net = False
            prediction, _ = inference(model, args, te_loader, 
                                          u2net, with_label=False)
        models_pred.append(prediction)

    def Ensemble(preds, major=2):
        ensemble_dict={}
        # Segmentation mask ensemble
        _mask = np.zeros_like(preds[0]['mask'])
        for m in range(len(preds)):
            _mask += preds[m]['mask']
        ensemble_dict['mask'] = (_mask >= major).astype(np.uint8)
        
        # Classification ensemble
        identity_mat = np.eye(3) # num of class
        pred = identity_mat[preds[m]['class']]
        for m in range(1, len(preds)):
            _pred = identity_mat[preds[m]['class']]
            pred += _pred
        ensemble_dict['class'] = pred.argmax(axis=1)
        return ensemble_dict
        
    ensemble_pred = Ensemble(models_pred)
    
    # Results
    run = wandb.init(project=args.project, name=args.exp_name,
                     job_type=args.dataset+'_test', config=args.config)
    
    dice_m1 = np.mean(U.calculate_score(ensemble_pred['mask'][:,0],
                                        label['mask'][:,0]))
    dice_m2 = np.mean(U.calculate_score(ensemble_pred['mask'][:,1],
                                        label['mask'][:,1]))
    dice_m3 = np.mean(U.calculate_score(ensemble_pred['mask'][:,2],
                                        label['mask'][:,2]))
    
    avg_score = np.mean([dice_m1, dice_m2, dice_m3])
    acc = metrics.accuracy_score(label['class'], ensemble_pred['class']) 
    
    results = wandb.Table(columns = ['dice_m1', 'dice_m2', 'dice_m3', 'avg', 'acc'])
    results.add_data(dice_m1, dice_m2, dice_m3, avg_score, acc)
    wandb.log({'Test results': results})
    run.finish()
    
    print('[Ensemble results] ===========================================')
    print('classification accuracy:', acc)
    print('avg_score: %.6f' %(avg_score))
    print('score_m1: %.6f' %(dice_m1))
    print('score_m2: %.6f' %(dice_m2))
    print('score_m3: %.6f' %(dice_m3))
    print('=====================================================')
    
    if args.vis_path:
        show_examples(models_pred, ensemble_pred, label, args.vis_path)
        
        
        
        
def cvt2class(_cl):
    if _cl == 0:
        _class = 'RCA'
    elif _cl == 1:
        _class = 'LAD'
    else:
        _class = 'LCX'
    return _class

def show_examples(models_pred, ensemble_pred, label, vis_path):
    for idx in range(len(label['mask'])):
        font=18
        fig = plt.figure(figsize=(20,25))
        gs = gridspec.GridSpec(5,4, width_ratios=[1]*4)
        ax, ax_num = [], 0

        # [0,0] Source image
        ax.append(fig.add_subplot(gs[ax_num]))
        ax[-1].imshow(label['img'][idx], cmap='gray')
        ax[-1].axis('off')
        ax[-1].title.set_text('Source')
        ax[-1].title.set_fontsize(font)
        ax[-1].text(0.5,-0.12, 'Class:'+cvt2class(label['class'][idx]), 
         fontsize=font, ha='center', transform=ax[-1].transAxes)

        # [0, 1] Label M1
        ax_num += 1
        ax.append(fig.add_subplot(gs[ax_num]))
        ax[-1].imshow(label['mask'][idx,0], cmap='gray')
        ax[-1].axis('off')
        ax[-1].title.set_text('Label(M1)')
        ax[-1].title.set_fontsize(font)

        # [0, 2] Label M2
        ax_num += 1
        ax.append(fig.add_subplot(gs[ax_num]))
        ax[-1].imshow(label['mask'][idx,1], cmap='gray')
        ax[-1].axis('off')
        ax[-1].title.set_text('Label(M2)')
        ax[-1].title.set_fontsize(font)

        # [0, 3] Label M3
        ax_num += 1
        ax.append(fig.add_subplot(gs[ax_num]))
        ax[-1].imshow(label['mask'][idx,2], cmap='gray')
        ax[-1].axis('off')
        ax[-1].title.set_text('Label(M3)')
        ax[-1].title.set_fontsize(font)

        for j, P in enumerate([ensemble_pred,    #ensemble
                                 models_pred[0],   #u2net
                                 models_pred[1],   #Unet++
                                 models_pred[2]]): #deeplab
            if j == 0:
                Network = 'Ensemble'
            elif j == 1:
                Network = 'U2Net'
            elif j == 2:
                Network = 'Unet++'
            else:
                Network = 'Deeplab'

            Score1 = U.calculate_score(label['mask'][idx,0], 
                                     P['mask'][idx,0],'Dice', batch=False)
            Score2 = U.calculate_score(label['mask'][idx,1], 
                                     P['mask'][idx,1],'Dice', batch=False)
            Score3 = U.calculate_score(label['mask'][idx,2], 
                                     P['mask'][idx,2],'Dice', batch=False)

            # [j+1, 0] Source image
            ax_num += 1
            ax.append(fig.add_subplot(gs[ax_num]))
            ax[-1].imshow(label['img'][idx], cmap='gray')
            ax[-1].axis('off')
            ax[-1].text(0.5,-0.12, Network + ':'+cvt2class(P['class'][idx]), 
                 fontsize=font, ha='center', transform=ax[-1].transAxes)

            # [j+1, 1] Prediction M1
            ax_num += 1
            over = overlay(label['img'][idx], 
                           label['mask'][idx,0],
                           P['mask'][idx,0])
            ax.append(fig.add_subplot(gs[ax_num]))
            ax[-1].imshow(over)
            ax[-1].axis('off')
            ax[-1].text(0.5,-0.12, 'F1 = {:.4f}'.format(Score1), 
                     fontsize=font, ha='center', transform=ax[-1].transAxes)

            # [j+1, 2] Prediction M2
            ax_num += 1
            over = overlay(label['img'][idx], 
                           label['mask'][idx,1],
                           P['mask'][idx,1])
            ax.append(fig.add_subplot(gs[ax_num]))
            ax[-1].imshow(over)
            ax[-1].axis('off')
            ax[-1].text(0.5,-0.12, 'F1 = {:.4f}'.format(Score2), 
                     fontsize=font, ha='center', transform=ax[-1].transAxes)

            # [j+1, 3] Prediction M3
            ax_num += 1
            over = overlay(label['img'][idx], 
                           label['mask'][idx,2],
                           P['mask'][idx,2])
            ax.append(fig.add_subplot(gs[ax_num]))
            ax[-1].imshow(over)
            ax[-1].axis('off')
            ax[-1].text(0.5,-0.12, 'F1 = {:.4f}'.format(Score3), 
                     fontsize=font, ha='center', transform=ax[-1].transAxes)

        fig.savefig(vis_path+str(idx)+'.png')
        plt.clf()
        plt.close('all')
            
def overlay(img, mask, P):
    Disp = np.zeros((512,512,3), dtype=np.uint8)
    Disp[:,:,0] = img
    Disp[:,:,1] = img
    Disp[:,:,2] = img
    
    Disp[:,:,0][P*mask==1] = 255   # TP
    Disp[:,:,1][P*mask==1] = 0
    Disp[:,:,2][P*mask==1] = 0
    
    Disp[:,:,0][P>mask] = 0   # FP
    Disp[:,:,1][P>mask] = 255
    Disp[:,:,2][P>mask] = 0
    
    Disp[:,:,0][P<mask] = 255   # FN
    Disp[:,:,1][P<mask] = 255
    Disp[:,:,2][P<mask] = 0
    return Disp        
   
    
        
if __name__ == '__main__':
    main()
    
    
    