#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import random
import pyonmttok
import logging
from collections import defaultdict
from tqdm import tqdm
import pyonmttok
import os

separ = '￨'
keep = '·'
space = '~'


def decode(line, get_tags=True):
    tuples = line.rstrip().split(' ')
    tuples = [t.split(separ) for t in tuples]
    text = ' '.join([t[0] for t in tuples])
    if get_tags:
        tags = ' '.join([separ.join(t[1:]) for t in tuples])
        return text, tags
    return text


def create_dataset(file, target_file):
    path_clean = os.path.abspath(target_file + '.fr')
    path_noise = os.path.abspath(target_file + '.noise.fr')
    path_tag = os.path.abspath(target_file + '.tag.fr')

    file_clean = open(path_clean, 'w')
    file_noise = open(path_noise, 'w')
    file_tag = open(path_tag, 'w')

    with open(file, 'r') as f:
        first = True
        tags = None
        for line in f:
            if line == '\n':
                first = True
            elif first:
                ref = decode(line, get_tags=False)
                first = False
            else:
                if tags is not None:
                    file_clean.write('\n')
                    file_noise.write('\n')
                    file_tag.write('\n')
                text, tags = decode(line)
                file_clean.write(ref)
                file_noise.write(text)
                file_tag.write(tags)

    file_clean.close()
    file_noise.close()
    file_tag.close()


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
    parser.add_argument('file', help="Input file/s")
    ### optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-log', default='info', help="Logging level [debug, info, warning, critical, error] (info)")
    parser.add_argument('-to', help="core file name (with path) corresponding to target dataset save files")
    args = parser.parse_args()
    create_logger('stderr', args.log)
    logging.info("generate dataset")

    create_dataset(args.file, args.to)
