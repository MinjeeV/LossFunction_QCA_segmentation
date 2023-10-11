#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A00_00 --model='U2net'

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A00_01 --model='UnetPP'

CUDA_VISIBLE_DEVICES=0,1,2,3 python initial.py A00_02 --model='Deeplab'
