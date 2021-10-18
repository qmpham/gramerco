#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import pyonmttok
import argparse
import sys
from Noise import Noise, Spacy
from collections import defaultdict
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', help="Input file/s", nargs='+')
    ### optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-max', '--max_tokens', type=int, default=3, help='Attempts to inject noise in this number of tokens per sentence (def 3)')
    parser.add_argument('-lex', '--lexicon_file', help='File with lexicon (one word per line with lemma,cgram,genre,numberm,morpho information)')
    parser.add_argument('-rep', '--replace_file', help='File with replacements (lines with space-separated words that can replace each other)')
    parser.add_argument('-app', '--append_file', help='File with words to be appended (one word per line)')
    parser.add_argument('--p_del', type=float, default=0.01, help='Prob of deleting one token (def 0.01)')
    parser.add_argument('--p_lex', type=float, default=0.05, help='Prob of transforming a token using morph features (def 0.05) [to use with -lex]')
    parser.add_argument('--p_app', type=float, default=0.05, help='Prob of appending one token (def 0.05) [to use with -app]')
    parser.add_argument('--p_rep', type=float, default=0.05, help='Prob of replacing one token (def 0.05) [to use with -rep]')
    parser.add_argument('--p_swa', type=float, default=0.01, help='Prob of swapping two tokens (def 0.01)')
    parser.add_argument('--p_cas', type=float, default=0.05, help='Prob of changing the first char case (def 0.05)')
    parser.add_argument('--p_mer', type=float, default=0.01, help='Prob of merging two tokens (def 0.01)')
    parser.add_argument('--p_spl', type=float, default=0.01, help='Prob of splitting one token (def 0.01)')
    parser.add_argument('--p_hyp', type=float, default=0.05, help='Prob of joining two tokens with an hyphen (def 0.05)')
    args = parser.parse_args()
    #t = pyonmttok.Tokenizer("aggressive") #https://github.com/OpenNMT/Tokenizer/tree/master/bindings/python
    n = Noise(args)
    s = Spacy()

    nsents = 0
    for f in args.files:
        with open(f) as fd:
            lines = []
            for l in fd:
                lines.append(l.rstrip())
        sys.stderr.write('READ {} sentences from {}\n'.format(len(lines),f))
        for i in tqdm(range(len(lines))):
            #l,_ = t.tokenize(lines[i])
            l = s.analyze(lines[i])
            n.add(l)
            nsents += 1
                
    sys.stderr.write('found {} sentences\n'.format(nsents))
    n.stats()
        
        
        
