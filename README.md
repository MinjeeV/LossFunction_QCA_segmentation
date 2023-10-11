# QCA_segmentation
## Main, LM+Main, LM+Main+Branch segmentation (+RCA, LAD, LCX classification)

### v.2023.10
기존의 Jupyter 환경 코드에서 Python 코드로 수정하면서, 여러 부분들을 수정.

* U2net의 경우, 기존에서 조금 수정. 
  1) output shape: (Class, mask1, mask2, ..., mask7) 순서에서 (masks, class)로 변경  *masks에 7개의 Supervision을 위한 mask 존재.
     -> Segmentation_models_pytorch에서 사용하는 순서와 일치시킴
  2) F.sigmoid에서 torch.sigmoid로 수정. (torch 권장 사항)



### Dataset
1. \\\192.168.45.199\Angiography\1. QCA\10. DatasetForDL 폴더에 Input image와 Mask가 저장되어있다.

2. meta.json 파일을 통해 Dataset을 불러온다.

   데이터 구성: {'Fold_idx': [fold1, fold2, fold3, fold4, fold5],
                 'meta': {'idx', 'id', 'class', 'lesion'}}

     fold1: fold1의 index list
   
     idx: 각 폴드의 index list를 통해 해당 이미지에 접근
   
     id: image와 mask를 load하기 위한 patient number
   
     class: 해당 데이터의 class / RCA(0), LAD(1), LCX(2)
   
     lesion: lesion points [x, y] (최대 5개) 


### Code
initial.py : 하나의 모델에 대해서 Train 및 Test. (CBN 및 CUH data test 진행)

evaluation_ensemble.py : 학습된 3개의 모델에 대해서 Ensemble진행 및 Visualization. 

execute.sh : scripts 내의 쉘스크립트를 순차적으로 실행시키는 스크립트.


