#!/bin/bash

source ~/anaconda3/bin/activate gramerco

DATA_NAME=AFP
DATA_DIR=../resources

mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/$DATA_NAME
mkdir -p $DATA_DIR/$DATA_NAME/$DATA_NAME-raw

wget http://www.lexique.org/databases/Lexique383/Lexique383.tsv -P $DATA_DIR
ln -sf /nfs/CORPUS/zhangd/AFP/raw/YEAR-MONTH/HEADLINE/HEADLINES*fr.txt $DATA_DIR/$DATA_NAME/$DATA_NAME-raw
ln -sf /nfs/CORPUS/zhangd/AFP/raw/YEAR-MONTH/BODY/BODY_*fr.txt $DATA_DIR/$DATA_NAME/$DATA_NAME-raw

python -m spacy download fr_core_news_md
