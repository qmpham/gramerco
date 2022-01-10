#!/bin/bash

source ~/anaconda3/bin/activate gramerco

DATA_DIR=../resources
DATA_NAME=AFP
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin
DATA_SRC=$DATA_DIR/$DATA_NAME/$DATA_NAME-noise/$DATA_NAME.test.noise.fr
DATA_TGT=$DATA_DIR/$DATA_NAME/$DATA_NAME-noise/$DATA_NAME.test.tag.fr

# DATA_NAME=bin
# DATA_BIN=$DATA_DIR/debug
# DATA_SRC=$DATA_BIN/debug.fr
# DATA_TGT=$DATA_BIN/debug.tag.fr

SAVE_PATH=$DATA_DIR/models/gramerco-fr
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tensorboard

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:50 \
python test_model.py \
      --data-bin $DATA_BIN/$DATA_NAME \
      --file-src $DATA_SRC \
      --file-tag $DATA_TGT  \
      --sample 1000000 \
      --log DEBUG \
      --model-iter -1 \
      --save $SAVE_PATH \
      --model-type decision \
      --model-id freeze30k+dropout0.1+ls0.2+beta0.2+cumul4+rdm0.5-decision2 \
      --lex $DATA_DIR/Lexique383.tsv \
      --app $DATA_DIR/AFP/AFP-lex/lexique.app \
      --tokenizer flaubert/flaubert_base_cased \
      --batch-size 20 \
      --ignore-clean \
      # --raw \
      # --gpu \
