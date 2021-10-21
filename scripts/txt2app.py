#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pyonmttok
from collections import defaultdict
from tqdm import tqdm

if __name__ == '__main__':
    t = pyonmttok.Tokenizer("conservative", joiner_annotate=False)
    nsents = 0
    ntokens = 0
    bigrams = defaultdict(int)
    unigrams = defaultdict(int)
    for l in sys.stdin:
        toks, _ = t.tokenize(l.rstrip())
        for i in range(len(toks)-1):
            if toks[i].isalpha() and toks[i+1].isalpha():
                bigrams[toks[i]+' '+toks[i+1]] += 1
            if toks[i].isalpha():
                unigrams[toks[i]] += 1
        if toks[i].isalpha():
            unigrams[toks[-1]] += 1
        nsents += 1
        ntokens += len(toks)
    sys.stderr.write('Found {} sentences with {} words. Vocab of bigrams is {}\n'.format(nsents, ntokens, len(bigram)))
    for w,n in sorted(bigram.items(), key=lambda kv: kv[1], reverse=True):
        print('{} {}'.format(w,n))
        
        
        
