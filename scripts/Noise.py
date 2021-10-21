#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import sys
import spacy
import logging
from collections import defaultdict
separ = '￨' ### separator for output
space = '~' ### replaces space by this on Spacy txt/lem tokens
keep = '·'

class Tok():
    def __init__(self, txt, lem='', pos='', inf=''):
        self.txt = txt
        self.lem = lem
        self.pos = pos
        self.inf = inf
        
class Spacy():
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_md")

    def analyze(self, l):
        toks = []
        for token in self.nlp(l.rstrip()):
            txt = token.text.replace(' ',space)
            lem = token.lemma_.replace(' ',space)
            pos = str(token.pos_)
            inf = str(token.morph).replace('|',separ)
            #token.tag_, #token.dep_, #token.shape_, #token.is_alpha, #token.is_stop
            toks.append(Tok(txt,lem=lem,pos=pos,inf=inf))
        return toks

class Lexicon():
    def __init__(self,f):
        self.mot2linf = defaultdict(set)
        self.lem2lmot = defaultdict(set)
        with open(f) as fd:
            for l in fd: 
                toks = l.rstrip().split('\t')
                mot, lemma, cgram, genre, nombre, infover = toks[0].replace(' ',space), toks[2].replace(' ',space), toks[3], toks[4], toks[5], toks[10].replace(':',separ)
                if not genre:
                    genre = '-'
                if not nombre:
                    nombre = '-'

                if cgram == 'VER' or cgram == 'AUX':
                    self.lem2lmot[lemma].add(mot)
                    for v in infover.split(';'):
                        if len(v):
                            inf = separ.join([cgram, genre, nombre, v])
                            self.mot2linf[mot].add(inf)
                    
                elif cgram == 'NOM' or  cgram == 'ADJ':
                    self.lem2lmot[lemma].add(mot)
                    inf = separ.join([cgram, genre, nombre])
                    self.mot2linf[mot].add(inf)

#                elif cgram == 'NOM' or  cgram == 'ADV' or cgram == 'PRE' or cgram == 'ONO' or cgram == 'CON' or cgram.startswith('PRO') or cgram.startswith('ADJ') or cgram.startswith('ART'):
#                    self.lem2lmot[lemma].append(mot)
#                    inf = [lemma, cgram, genre, nombre]
#                    #print('{}\t{}'.format(mot,inf))
#                    self.mot2linf[mot].append(inf)

        self.lmot = list(self.mot2linf.keys())
        logging.info('READ {} words and {} lemmas from {}'.format(len(self.mot2linf), len(self.lem2lmot), f))

class Noise():
    def __init__(self,args):
        self.args = args
        self.counts = defaultdict(int)
        self.l = Lexicon(args.lex)


    def noise(self,t):
        if self.args.adj and t.pos == 'ADJ' or self.args.nom and t.pos == 'NOUN' or self.args.ver and (t.pos == 'VERB' or t.pos == 'AUX'):
            logging.debug('000 {} {} {}'.format(t.txt,t.pos,t.inf))
            txt_truecase = t.txt
            t.txt = t.txt.lower() ### all text tokens appear lowercased in Lexicon
            if t.lem not in self.l.lem2lmot:
                return '', ''
            logging.debug('111')
            if t.txt not in self.l.mot2linf:
                return '', ''
            logging.debug('222')
            ltxt = list(self.l.lem2lmot[t.lem]) ### set of words with same lemma lem
            if t.txt in ltxt:
                ltxt.remove(t.txt)
            if len(ltxt) == 0:
                return '', ''
            logging.debug('333')
            ltxt.append(txt_truecase) ### last is current token as it originally appears (not lowercsed)
            linf = list(self.l.mot2linf[t.txt]) ### list of inf's associated to word txt
            tag = self.get_lexicon_tag(t,linf) ### returns the inf in linf that corresponds to token t
            logging.debug('444 {} {}'.format(tag,linf))
            if tag not in linf:
                return '', ''
            logging.debug('555 {}'.format(separ.join(list(ltxt))))
            self.counts[tag] += 1
            return separ.join(list(ltxt)), tag
        return '', ''

    def get_lexicon_tag(self, t, linf): 
        spacy_txt = t.txt #saisit
        spacy_lem = t.lem #saisir
        spacy_pos = t.pos #VERB
        spacy_inf = t.inf #Mood=Ind￨Number=Sing￨Person=3￨Tense=Past￨VerbForm=Fin
        inf_curr = []

        ##############
        ### pos ######
        ##############
        if spacy_pos == 'VERB': 
            pos = 'VER'
        elif spacy_pos == 'AUX': 
            pos = 'AUX'
        elif spacy_pos == 'ADJ': 
            pos = 'ADJ'
        elif spacy_pos == 'NOUN': 
            pos = 'NOM'
        else:
            logging.error('invalid pos={}\n'.format(spacy_pos))
            sys.exit()
        inf_curr.append(pos)

        ### if Lexicon has a single entry for the given curr_txt with same pos i just consider it as the good one to return
        linf_i = self.single_entry_with_pos(pos,linf)
        if linf_i: ### for any pos
            return linf_i

        ##############
        ### genre ####
        ##############
        if 'Gender=Fem' in spacy_inf: genre='f'
        elif 'Gender=Masc' in spacy_inf: genre='m'
        else: genre='-'
        inf_curr.append(genre)

        ##############
        ### nombre ###
        ##############
        if 'Number=Sing' in spacy_inf: nombre='s'
        elif 'Number=Plur' in spacy_inf: nombre='p'
        else: nombre='-'
        inf_curr.append(nombre)    

        ### if Lexicon has a single entry for pos's 'ADJ' or 'POS' for the given curr_txt i just consider it as the good one to return
        if pos == 'ADJ' or pos == 'NOM':
            linf_i = self.single_entry_with_pos(pos,linf)
            if linf_i:
                return linf_i

        if pos != 'VER' and pos != 'AUX':
            return separ.join(inf_curr) #[pos, genre, nombre]

        ##############
        ### vinf #####
        ##############

        features = []
        if 'VerbForm=Inf' in spacy_inf:
            features.append('inf')
        elif 'VerbForm=Fin' in spacy_inf:
            features.append('fin')
        elif 'VerbForm=Part' in spacy_inf:
            features.append('par')

        if 'Tense=Past' in spacy_inf:
            features.append('pas')
        elif 'Tense=Pres' in spacy_inf:
            features.append('pre')
        elif 'Tense=Fut' in spacy_inf:
            features.append('fut')
        elif 'Tense=Imp' in spacy_inf:
            features.append('imp')

        if 'Mood=Ind' in spacy_inf:
            features.append('ind')
        elif 'Mood=Sub' in spacy_inf:
            features.append('sub')

        person = number = ''
        if 'Person=1' in spacy_inf:
            person = '1'
        elif 'Person=2' in spacy_inf:
            person = '2'
        elif 'Person=3' in spacy_inf:
            person = '3'
        if 'Number=Sing' in spacy_inf:
            number = 's'
        elif 'Number=Plur' in spacy_inf:
            number = 'p'
    
        if person and number:
            features.append(person+number)

        linf_i = self.single_entry_with_infs(features, linf)
        if linf_i:
            return linf_i

        return ''

    def single_entry_with_pos(self,pos,linf):
        entries = [i for i in range(len(linf)) if linf[i].split(separ)[0] == pos] 
        if len(entries) == 1:
            return linf[entries[0]]
        return ''

    def single_entry_with_infs(self,feats,linf): ### feats is a list of features like : ['fut', 'inf', '3s', ...]
        entries = []
        for feat in feats: ### check if this feat reduces to one the number of possible inf's in linf
            n_found_infs_in_linf_with_feat = 0
            for inf in linf:
                if feat in inf.split(separ):
                    n_found_infs_in_linf_with_feat += 1
                    found_inf = inf
            if n_found_infs_in_linf_with_feat == 1:
                return found_inf
        return ''

    def stats(self):
        logging.info('Vocab of {} tags'.format(len(self.counts)))
        for k,v in sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True):
            logging.info('{}\t{}'.format(k,v))

        
