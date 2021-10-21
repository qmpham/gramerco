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
    dic = defaultdict(float)
    for l in sys.stdin:
        toks, _ = t.tokenize(l.rstrip())
        for tok in toks:
            dic[tok] += 1
        nsents += 1
        ntokens += len(toks)
    sys.stderr.write('Found {} sentences with {} words. Vocab is {}\n'.format(nsents, ntokens, len(dic)))
    for w,n in sorted(dic.items(), key=lambda kv: kv[1], reverse=True):
        print('{} {:.8f}'.format(w,n/ntokens))
        
        
        
