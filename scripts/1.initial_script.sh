#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A00_00_U2net --model='U2net'

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A00_01_UnetPP --model='UnetPP'

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A00_02_Deeplab --model='Deeplab'


# use mixed precision method

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A01_00_U2net --model='U2net' --use_fp16=1 --batch_size=24

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A01_01_UnetPP --model='UnetPP' --use_fp16=1 --batch_size=24

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A01_02_Deeplab --model='Deeplab' --use_fp16=1 --batch_size=24


# use mixed precision method

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A01_03_U2net --model='U2net' --use_fp16=4 --batch_size=24

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A01_04_UnetPP --model='UnetPP' --use_fp16=4 --batch_size=24

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A01_05_Deeplab --model='Deeplab' --use_fp16=4 --batch_size=24
