#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import logging
from add_french_noise import create_logger, read_rep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lex', help='Path of Lexique383.tsv')
    parser.add_argument('-art', action='store_true', help="Articles")
    parser.add_argument('-pre', action='store_true', help="Prepositions")
    parser.add_argument('-pro', action='store_true', help="Pronouns")
    parser.add_argument('-adv', action='store_true', help="Adverbs")
    parser.add_argument('-pun', action='store_true', help="Punctuation")
    parser.add_argument('-log', default='info', help="Logging level [debug, info, warning, critical, error] (info)")
    args = parser.parse_args()
    create_logger('stderr',args.log)
    punctuation = [',', '.', ':', ';', '\'', '"', '!', '?', '<', '>', '(', ')', '-']
    rep = read_rep(args.lex)
    pos2mot = rep['pos2mot']
    for pos in pos2mot:
        if pos != 'ART' and pos != 'PRO' and pos != 'PRE' and pos != 'ADV':
            continiue
        if pos == 'ART' and not args.art:
            continue
        elif pos == 'PRE' and not args.pre:
            continue
        elif pos == 'PRO' and not args.pro:
            continue
        elif pos == 'ADV' and not args.adv:
            continue
        for mot in pos2mot[pos]:
            print(mot)
    if args.pun:
        print('\n'.join(punctuation))

        
        
