#!/bin/bash

source ~/anaconda3/bin/activate gramerco

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin

SAVE_PATH=$DATA_DIR/models/gramerco-fr
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tensorboard

CUDA_VISIBLE_DEVICES=0,1 \
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:50 \
python infer.py --text "Le colonel Frantz a déstabilisera le pays entier." \
      --log DEBUG \
      --save $SAVE_PATH \
      --model-id 1638781851 \
      --lex $DATA_DIR/Lexique383.tsv \
      --app $DATA_DIR/$DATA_NAME/$DATA_NAME-lex/lexique.app \
      --tokenizer flaubert/flaubert_base_cased \
      --batch-size 20 \
      --gpu \
      --gpu-id 1
