#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from Noise import Spacy, Lexicon, Noise
from collections import defaultdict
from tqdm import tqdm
import logging

def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.info('Created Logger level={}'.format(loglevel))
    else:
        logging.basicConfig(filename=logfile, format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.info('Created Logger level={} file={}'.format(loglevel, logfile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lex', help='Path of Lexique383.tsv')
    parser.add_argument('files', help="Input file/s", nargs='+')
    parser.add_argument('-log', default='info', help="Logging level [debug, info, warning, critical, error] (info)")
    args = parser.parse_args()
    create_logger('stderr',args.log)

    s = Spacy()
    n = Noise(args)
    nsents = 0
    ntoks = 0
    for f in args.files:
        with open(f) as fd:
            lines = fd.read().splitlines()
        logging.info('READ {} sentences from {}'.format(len(lines),f))
        for l in tqdm(lines):
            logging.debug('--- {}'.format(l))
            toks = s.analyze(l)
            logging.debug(s.compose(toks))
            out = [l]
            for t in toks:
                txt_other, tag_curr = n.noise(t)
                if txt_other and tag_curr:
                    out.append(tag_curr+':'+txt_other)
            print('\t'.join(out))
            nsents += 1
            ntoks += len(toks)
        logging.info('Found {} sentences with {} tokens'.format(nsents,ntoks))
        n.stats()
        
        
        
