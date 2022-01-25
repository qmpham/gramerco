#!/bin/bash

source ~/anaconda3/bin/activate gramerco

MOSES=/nfs/RESEARCH/bouthors/packages/mosesdecoder/scripts
DETOKENIZER=$MOSES/tokenizer/detokenizer.perl

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin

DATA_SRC=$DATA_DIR/dictates/test-dictates/dicts.err

SAVE_PATH=$DATA_DIR/models/gramerco-fr
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tensorboard

CUDA_VISIBLE_DEVICES=0,1 \
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:50 \
python infer.py \
      --file $DATA_SRC \
      --log DEBUG \
      --save $SAVE_PATH \
      --model-id freeze30k+dropout0.1+ls0.2+beta0.2+cumul4+rdm0.5-decision2 \
      --lex $DATA_DIR/Lexique383.tsv \
      --app $DATA_DIR/$DATA_NAME/$DATA_NAME-lex/lexique.app \
      --tokenizer flaubert/flaubert_base_cased \
      --batch-size 20 \
      | perl $DETOKENIZER -l fr -q \
      | sed -r 's/# # #/###/g' \
      | sed -z 's/\n\n/\n/g' \
      > $DATA_DIR/dictates/generated-dictates/dicts.infer.cor
      # --gpu \
      # --gpu-id 1
