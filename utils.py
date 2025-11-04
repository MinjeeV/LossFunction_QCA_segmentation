import torch
import torch.nn as nn
from pytorch_model_summary import summary
import sklearn.metrics as metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl

from skimage.morphology import remove_small_objects, remove_small_holes
from skimage import measure
from skimage.measure import label

def model_summary(model, in_chans=1, size=512, show_input=False):
    print(summary(model, torch.zeros((1,in_chans,size,size)), show_input=show_input))

def set_foldNUM(i, k):
    fold = list(range(k))
    val = fold[(i+k-2)%k]
    te = fold[(i+k-1)%k]
    fold.remove(val)
    fold.remove(te)
    return fold, val, te

def select_fold(i, folds, k=5):
    tr, val, te = set_foldNUM(i, k)
    
    tr_data = []
    for f in tr:
        tr_data += folds[f]
        
    val_data = folds[val]
    te_data = folds[te]
    return tr_data, val_data, te_data


'''
=================================
Evaluation metric
=================================
'''
# Jaccard, Dice, Recall, Precision, Accuracy, CLD
def jaccard(y_true, y_pred, smooth=1e-08):
    product = np.multiply(y_true, y_pred)
    intersection = np.sum(product)
    union = np.sum(y_true) + np.sum(y_pred)
    return (intersection + smooth) / (union - intersection + smooth)

def dice_coef(y_true, y_pred, smooth=1e-08):
    product = np.multiply(y_true, y_pred)
    intersection = np.sum(product)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def recall(y_true, y_pred, smooth=1e-08): # sensitivity
    Y_t = y_true
    P_t = y_pred
    P_f = np.ones(P_t.shape, dtype=np.uint8) - P_t
    
    TP = Y_t * P_t
    FN = Y_t * P_f
    return (np.sum(TP) + smooth) / (np.sum(TP+FN) + smooth)

def precision(y_true, y_pred, smooth=1e-08):
    Y_t = y_true
    Y_f = np.ones(Y_t.shape, dtype=np.uint8) - Y_t
    P_t = y_pred
    
    TP = Y_t * P_t
    FP = Y_f * P_t
    return (np.sum(TP) + smooth) / (np.sum(TP+FP) + smooth)

def accuracy(y_true, y_pred, smooth=1e-08):
    Y_t = y_true
    Y_f = np.ones(Y_t.shape, dtype=np.uint8) - Y_t
    P_t = y_pred
    P_f = np.ones(P_t.shape, dtype=np.uint8) - P_t
    
    TP = Y_t * P_t
    FN = Y_t * P_f
    TN = Y_f * P_f
    FP = Y_f * P_t
    return (np.sum(TP+TN) + smooth) / (np.sum(TP+TN+FP+FN) + smooth)

def CLD(ts, ps):
    ts_c = np.argwhere(ts > 0)
    ps_c = np.argwhere(ps > 0)
    n = len(ps_c)
    if n == 0:
        return 0
    dist_sum = 0
    for i in range(n):
        dist_sum += np.amin(distance.cdist(ps_c[i:i+1], ts_c, 'euclidean'))
    return dist_sum / n


def calculate_score(y_true, y_pred, func='Dice', batch=True):
    if func == 'Dice':
        f = dice_coef
    elif func == 'jaccard':
        f = jaccard
    elif func == 'accuracy':
        f = accuracy
    elif func == 'recall':
        f = recall
    elif func == 'precision':
        f = precision
    elif func == 'CLD':
        f = CLD
    
    if batch:
        score = np.zeros(len(y_pred), dtype=np.float32)
        for i in range(len(y_pred)):
            score[i] = f(y_true[i], y_pred[i])
        return score
    else:
        return f(y_true, y_pred)

class Dice_Coef_Loss():
    
    def dice_coef(self, y_true, y_pred, smooth=1e-08):
        batch_size = y_true.shape[0]
        y_pred = torch.clamp(y_pred, 0, 1)
        y_true_f = y_true.view(batch_size, -1)
        y_pred_f = y_pred.view(batch_size, -1)
        intersection = torch.sum(y_true_f*y_pred_f, axis=-1)
        union = torch.sum(y_true_f, axis=-1) + torch.sum(y_pred_f, axis=-1)
        return torch.mean((2. * intersection + smooth) / (union + smooth))
    def dice_coef_loss(self, y_true, y_pred): 
        return 1 - self.dice_coef(y_true, y_pred)
    
    def loss(self, predict, target):
        return self.dice_coef_loss(target, predict)
    
class DiceLoss(nn.Module): # Re-implementation
    def __init__(self, smooth=1e-6, p=1, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class Supv_DiceLoss(nn.Module): # U2Net supervision dice loss
    def __init__(self, n=7):
        super(Supv_DiceLoss, self).__init__()
        self.n = n
        self.dice = DiceLoss() #for memory efficiency
    def forward(self, predict, target):
        total_loss = 0
        for i in range(self.n):
            total_loss += self.dice(predict[i], target)
        return total_loss
        
'''
=================================
QCA post-processing
=================================
'''
def leave_biggest_only(img):
    labels, ncomponents = label(img, connectivity=1, return_num=True)
    max_id = None
    max_size = None
    if ncomponents == 1:
        return img
    for j in range(ncomponents):
        obj = (labels == (j+1)) * 1
        obj_size = np.sum(obj)
        if max_id is None:
            max_id = j
            max_size = obj_size
        elif obj_size > max_size:
            max_id = j
            max_size = obj_size
    return (labels == (max_id+1)) * 1
def post_processing(P, thres_prop=0, ch = 3):
    post_P = np.zeros_like(P, dtype=np.uint8)
    for i in range(len(P)):
        for j in range(ch):
            img = P[i,j,...].copy()
            img = remove_small_holes(img)
            if thres_prop == 0:
                post_P[i,j,...] = img
                continue
            img_size = np.sum(img)
            thres = img_size*thres_prop
            img = remove_small_objects(img>0, min_size=thres, 
                                       connectivity=1, in_place=False)*1
            if np.sum(img) > 0:
                img = leave_biggest_only(img)
            post_P[i,j,...] = img
    return post_P

    
'''
=================================
Classification report
=================================
'''

def print_confusion_matrix(confusion_matrix, class_names,title, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    df_cm=df_cm.reindex(columns=[class_names[1],class_names[0]],index=[class_names[1],class_names[0]])
    fig = plt.figure(figsize=figsize)
    
    try:
        heatmap = sns.heatmap(
                                df_cm, annot=True, fmt="d",cmap="RdPu", xticklabels=True, yticklabels=True
#                               ,annot_kws={'font':'serif'},
                              )
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    font_setting0 = mpl.font_manager.FontProperties()
#     font_setting0.set_family('serif') # 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
    font_setting0.set_size(20)
    font_setting0.set_style('normal') # 'normal', 'oblique', 'italic'
    font_setting0.set_weight('bold')
    
    plt.title(f'{title}', fontproperties=font_setting0)
    
    return fig

def get_classification_report(y_true, y_pred, category=['X', 'O']):
    report = metrics.classification_report(y_true, y_pred, target_names=category, output_dict=True,zero_division=0,digits=4)
    df_classification_report = pd.DataFrame(report).transpose()
    return df_classification_report

def get_confusion_matrix(y_true, y_pred, num_classes=1):
    if type(y_true)==list or type(y_pred) == list:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    assert (type(y_true) and type(y_pred)) == np.ndarray, 'y_true and y_pred must be numpy array type!'
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy=[]
    sensitivity=[]
    specificity=[]
    for i in range(num_classes):
        acc = metrics.accuracy_score(y_true,y_pred)
        accuracy.append(acc)
        
        cm = metrics.confusion_matrix(y_true,y_pred)
        total=sum(sum(cm))
        #####from confusion matrix calculate accuracy
        sens = cm[1,1]/(cm[1,0]+cm[1,1])
        sensitivity.append(sens)
        
        spec = cm[0,0]/(cm[0,0]+cm[0,1])
        specificity.append(spec)
        
    return cm, np.array(accuracy), np.array(sensitivity), np.array(specificity)
