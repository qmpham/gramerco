#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict
from Noise import separ, space

###cgram's:
#  64929 VER
#  48287 NOM
#  26806 ADJ
#   1841 ADV
#    236 ONO
#    123 ADJ:num
#     88 AUX
#     80 PRE
#     53 PRO:per
#     44 PRO:ind
#     36 ADJ:ind
#     35 CON
#     31 ADJ:pos
#     23 PRO:pos
#     17 PRO:int
#     17 PRO:dem
#     17 PRO:rel
#     10 ART:def
#      4 ADJ:dem
#      4 ADJ:int
#      4 ART:ind
#      1 LIA

###fields:
#ortho	phon	lemme	cgram	genre	nombre	freqlemfilms2	freqlemlivres	freqfilms2	freqlivres	infover	nbhomogr	nbhomoph	islem	nblettres	nbphons	cvcv	p_cvcv	voisorth	voisphon	puorth	puphon	syll	nbsyll	cv-cv	orthrenv	phonrenv	orthosyll	cgramortho	deflem	defobs	old20	pld20	morphoder	nbmorph

for l in sys.stdin: 
    toks = l.rstrip().split('\t')
    mot = toks[0]
    if ' ' in mot:
        mot = mot.replace(' ',space)
    lemma = toks[2]
    if ' ' in lemma:
        lemma = lemma.replace(' ',space)
    cgram = toks[3]
    genre = toks[4]
    nombre = toks[5]
    infover = toks[10]
    if not genre:
        genre = '-'
    if not nombre:
        nombre = '-'

    if cgram == 'VER' or cgram == 'AUX':
        for v in infover.split(';'):
            if len(v):
                val = separ.join([lemma, cgram, genre, nombre, v])
                print('{}\t{}'.format(mot,val))
                    
    elif cgram == 'NOM' or cgram.startswith('ADJ') or cgram.startswith('ART') or  cgram == 'ADV' or cgram == 'PRE' or cgram == 'ONO' or cgram == 'CON' or cgram.startswith('PRO'):
        val = separ.join([lemma, cgram, genre, nombre])
        print('{}\t{}'.format(mot,val))

    else:
        sys.stderr.write(l+'\n')

