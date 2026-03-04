#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


python evaluate.py \
    --data_path       ./data \
    --train_split     ./data/splits/train_semeval_parids-labels.csv \
    --val_split       ./data/splits/dev_semeval_parids-labels.csv \
    --preds      ./predictions/local_test.txt \
    --checkpoint ./checkpoints_bckp/svm_mxbai_20260304_072053.joblib \
    --categories_file ./data/dontpatronizeme_categories.tsv \
    --embedding_model mxbai