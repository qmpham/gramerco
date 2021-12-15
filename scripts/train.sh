#!/bin/bash

source ~/anaconda3/bin/activate gramerco

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin

SAVE_PATH=$DATA_DIR/models/gramerco-fr
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tensorboard


CUDA_VISIBLE_DEVICES=1 \
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:200 \
python train.py $DATA_BIN/$DATA_NAME \
      --log DEBUG \
      --model-id freeze20k+dropout0.1+ls0.2 \
      --model-type decision2 \
      --save $SAVE_PATH \
      --continue-from none \
      --lex $DATA_DIR/Lexique383.tsv \
      --app $DATA_DIR/$DATA_NAME/$DATA_NAME-lex/lexique.app \
      --tokenizer flaubert/flaubert_base_cased \
      --num-workers 10 \
      --tensorboard \
      -lang fr \
      --max-tokens 4096 \
      --max-sentences 128 \
      --required-batch-size-multiple 8 \
      --max-positions 510 \
      --n-epochs 1 \
      -lr 0.00001 \
      -ls 0.2 \
      --dropout 0.1 \
      --valid-iter 5000  \
      --early-stopping 10 \
      --gpu \
      --ignore-clean \
      --freeze-encoder 20000 \
      --grad-cumul-iter 1 \
      --random-keep-mask 0 \
      # --valid \
      # --test \
