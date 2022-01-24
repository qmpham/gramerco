#!/bin/bash

source ~/anaconda3/bin/activate gramerco

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin-2

SAVE_PATH=$DATA_DIR/models/gramerco-two-fr
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tensorboard


CUDA_VISIBLE_DEVICES=1 \
CUDA_LAUNCH_BLOCKING=1 \
python train_two.py $DATA_BIN/$DATA_NAME \
      --log DEBUG \
      --model-id freeze20k+ls0.2+cumul4+rdm0.5 \
      --save $SAVE_PATH \
      --continue-from none \
      --lex $DATA_DIR/Lexique383.tsv \
      --voc $DATA_DIR/common/french.dic.20k \
      --tokenizer flaubert/flaubert_base_cased \
      --num-workers 15 \
      --tensorboard \
      -lang fr \
      --max-tokens 4096 \
      --max-sentences 128 \
      --required-batch-size-multiple 1 \
      --min-positions 5 \
      --max-positions 510 \
      --n-epochs 1 \
      -lr 0.00001 \
      -ls 0.2 \
      --valid-iter 5000  \
      --early-stopping 10 \
      --ignore-clean \
      --freeze-encoder 20000 \
      --grad-cumul-iter 4 \
      --random-keep-mask 0.5 \
      --valid \
      --test \
      --gpu \
