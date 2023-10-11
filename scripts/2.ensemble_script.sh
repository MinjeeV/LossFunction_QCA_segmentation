#!/bin/bash

# A00_Ensemble
CUDA_VISIBLE_DEVICES=0,1 python evaluation_ensemble.py A00_Ensemble_CBN --model_list='A00_00_U2net, A00_01_UnetPP, A00_02_Deeplab' --vis_path='/CBN/Angiography/1. QCA/1-1. QCA 분석파일/QCA_RESULTS/TripleLabel_results/A00_Ensemble/' --dataset='CBN'

CUDA_VISIBLE_DEVICES=0,1 python evaluation_ensemble.py A00_Ensemble_CUH --model_list='A00_00_U2net, A00_01_UnetPP, A00_02_Deeplab' --dataset='CUH'


# A01_Ensemble
CUDA_VISIBLE_DEVICES=0,1 python evaluation_ensemble.py A01_Ensemble_CBN --model_list='A01_00_U2net, A01_01_UnetPP, A01_02_Deeplab' --dataset='CBN'

CUDA_VISIBLE_DEVICES=0,1 python evaluation_ensemble.py A01_Ensemble_CUH --model_list='A01_00_U2net, A01_01_UnetPP, A01_02_Deeplab' --dataset='CUH'


# A01_Ensemble
CUDA_VISIBLE_DEVICES=0,1 python evaluation_ensemble.py A01_Ensemble2_CBN --model_list='A01_03_U2net, A01_04_UnetPP, A01_05_Deeplab' --dataset='CBN'

CUDA_VISIBLE_DEVICES=0,1 python evaluation_ensemble.py A01_Ensemble_CUH --model_list='A01_03_U2net, A01_04_UnetPP, A01_05_Deeplab' --dataset='CUH'
