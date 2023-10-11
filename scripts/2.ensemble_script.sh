#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python evaluation_ensemble.py A00_Ensemble --vis_path='/CBN/Angiography/1. QCA/1-1. QCA 분석파일/QCA_RESULTS/TripleLabel_results/A00_Ensemble/' --dataset='CBN'

CUDA_VISIBLE_DEVICES=0,1 python evaluation_ensemble.py A00_Ensemble --dataset='CUH'
