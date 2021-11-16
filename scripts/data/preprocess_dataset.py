#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import random
import logging
from collections import defaultdict
from tqdm import tqdm
import torch
import os
import numpy as np
pwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(pwd))
sys.path.append(os.path.dirname(os.path.abspath(pwd)))
from tag_encoder import TagEncoder
from transformers import FlaubertTokenizer
import re


def partition(length: int, dev_size: float, test_size: float):
    assert dev_size + test_size < 1.
    idxs = np.arange(length)
    np.random.shuffle(idxs)

    idx_1 = int(test_size * length)
    idx_2 = idx_1 + int(dev_size * length)

    return {
            "train":    idxs[idx_2:],
            "dev":      idxs[idx_1:idx_2],
            "test":     idxs[:idx_1]
            }


def save_format(path, ids, mask):
    data = np.stack((ids, mask))
    shape = data.shape
    dtype = data.dtype
    metadata = str(shape) + '@' + str(dtype)
    metadata = metadata.ljust(50)
    logging.debug('saving metadata of ' + os.path.basename(path) + ':' + metadata + '#' + str(len(metadata)))
    with open(path, 'wb') as f:
        f.write(bytes(metadata, 'utf-8') + data.tobytes())


def preprocess_txt_file(file, target_folder, tokenizer, target_filename, split, noise="", split_indices=None):
    path = os.path.join(target_folder, os.path.basename(target_filename) + '.{}{}.fr.bin')

    with open(file, 'r') as f:
        txt_list = f.read().split('\n')
        data = tokenizer(
            txt_list,
            return_tensors="np",
            padding=True
        )
        logging.debug("data after tokenization")
        logging.debug(data)

    if split_indices is not None:
        save_format(path.format("train", noise), data['input_ids'][split_idxs["train"]], data['attention_mask'][split_idxs["train"]])
        save_format(path.format("dev", noise), data['input_ids'][split_idxs["dev"]], data['attention_mask'][split_idxs["dev"]])
        save_format(path.format("test", noise), data['input_ids'][split_idxs["test"]], data['attention_mask'][split_idxs["test"]])
    else:
        save_format(path.format(split, noise), data['input_ids'], data['attention_mask'])


def preprocess_tag_file(file, target_folder, target_filename, path_to_lex, path_to_app, split, split_indices=None):
    path = os.path.join(target_folder, os.path.basename(target_filename) + '.{}.tag.fr.bin')

    tag_encoder = TagEncoder()

    with open(file, 'r') as f:
        txt_list = f.read().split('\n')

    lens = [len(t.split(' ')) for t in txt_list]
    max_len = max(lens)
    ids = np.empty((len(txt_list), max_len), dtype=np.int64)
    mask = np.zeros((len(txt_list), max_len), dtype=np.int64)

    for i in range(len(txt_list)):
        ids[i, :lens[i]] = np.array([tag_encoder.tag_to_id(tag) for tag in txt_list[i].split(' ')], dtype=np.int64)
        mask[i, :lens[i]] = 1

    if split_indices is not None:
        save_format(path.format("train"), ids[split_idxs["train"]], mask[split_idxs["train"]])
        save_format(path.format("dev"), ids[split_idxs["dev"]], mask[split_idxs["dev"]])
        save_format(path.format("test"), ids[split_idxs["test"]], mask[split_idxs["test"]])
    else:
        save_format(path.format(split), idxs, mask)


def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
    else:
        logging.basicConfig(filename=logfile, format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Input files core name")
    ### optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-log', default='info', help="Logging level [debug, info, warning, critical, error] (info)")
    parser.add_argument('-to', default='', help="destination folder")
    parser.add_argument('-split', default='train', help="train, test or dev")
    parser.add_argument('-lex', default="/home/bouthors/workspace/gramerco-repo/gramerco/resources/Lexique383.tsv", help="Path to the Lexique table")
    parser.add_argument('-app', default="/home/bouthors/workspace/gramerco-repo/gramerco/resources/lexique.app", help="Path to the appendable word list")
    args = parser.parse_args()
    if not args.to:
        args.to = os.path.join(os.path.dirname(args.file), 'bin')
        try:
            os.mkdir(args.to)
        except: ...
    create_logger('stderr', args.log)

    logging.info("preprocess dataset")

    tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")

    if args.split == "all":
        f = open(args.file + '.tag.fr', 'r')
        length = len(f.readlines())
        f.close()
        split_idxs = partition(length, dev_size=0.1, test_size=0.1)
    else:
        split_idxs = None


    preprocess_txt_file(args.file + '.fr', args.to, tokenizer, args.file, args.split, noise="", split_indices=split_idxs)
    preprocess_txt_file(args.file + '.noise.fr', args.to, tokenizer, args.file, args.split, noise=".noise", split_indices=split_idxs)
    preprocess_tag_file(args.file + '.tag.fr', args.to, args.file, args.lex, args.app, args.split, split_indices=split_idxs)
