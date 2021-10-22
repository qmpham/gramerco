#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import logging
import pyonmttok
from add_french_noise import create_logger
from collections import defaultdict
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-minf', '--min_freq', type=int, default=1, help='Minimum frequency to keep a word (def 1)')
    parser.add_argument('-log', default='info', help="Logging level [debug, info, warning, critical, error] (info)")
    args = parser.parse_args()
    create_logger('stderr',args.log)

    t = pyonmttok.Tokenizer("conservative", joiner_annotate=False)
    nsents = 0
    ntokens = 0
    dic = defaultdict(float)
    for l in sys.stdin:
        toks, _ = t.tokenize(l.rstrip())
        for tok in toks:
            dic[tok] += 1
        nsents += 1
        ntokens += len(toks)
    sys.stderr.write('Found {} sentences with {} words. Total vocab is {}\n'.format(nsents, ntokens, len(dic)))
    N = 0
    for w,n in sorted(dic.items(), key=lambda kv: kv[1], reverse=True):
        if n < args.min_freq:
            break
        print('{} {}'.format(w,n/ntokens))
        N += 1
    sys.stderr.write('Output vocab is {}\n'.format(N))        
        
        
