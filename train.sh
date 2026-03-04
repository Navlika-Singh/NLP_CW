#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python train.py \
  --data_path ./data \
  --train_split ./data/splits/train_semeval_parids-labels.csv \
  --val_split   ./data/splits/dev_semeval_parids-labels.csv \
  --classifier  svm \
  --embedding_model mxbai