#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import sys
import spacy
from collections import defaultdict
separ = '￨' ### separator for output
space = '~' ### replaces space by this on Spacy txt/lem tokens
keep = '·'

################################################################################################
################################################################################################
################################################################################################

class Tok():
    def __init__(self, txt, original, lem='', pos='', inf='', tag=''):
        self.txt = txt
        self.original = original
        self.lem = lem
        self.pos = pos
        self.inf = inf
        self.tag = tag

    def modify(self, txt, lem= '', pos='', inf='', tag=''):
        self.txt = txt
        self.original = False
        self.lem = lem
        self.pos = pos
        self.inf = inf
        self.tag = tag
        
class Spacy():
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_md")

    def analyze(self, l):
        toks = []
        for token in self.nlp(l.rstrip()):
            txt = token.text.replace(' ',space)
            lem = token.lemma_.replace(' ',space)
            pos = str(token.pos_)
            inf = str(token.morph).replace(':',separ)
            #token.tag_, #token.dep_, #token.shape_, #token.is_alpha, #token.is_stop
            toks.append(Tok(txt,True,lem=lem,pos=pos,inf=inf,tag=keep))
        return toks

class Lexicon():
    def __init__(self,f):
        self.mot2linf = defaultdict(list)
        self.lem2lmot = defaultdict(list)
        with open(f) as fd:
            for l in fd: 
                toks = l.rstrip().split('\t')
                mot, lemma, cgram, genre, nombre, infover = toks[0].replace(' ',space), toks[2].replace(' ',space), toks[3], toks[4], toks[5], toks[10]
                if not genre:
                    genre = '-'
                if not nombre:
                    nombre = '-'

                if cgram == 'VER' or cgram == 'AUX':
                    for v in infover.split(';'):
                        if len(v):
                            val = [lemma, cgram, genre, nombre, v]
                            #print('{}\t{}'.format(mot,val))
                            self.mot2linf[mot].append(val)
                            self.lem2lmot[lemma].append(mot)
                    
                elif cgram == 'NOM' or  cgram == 'ADV' or cgram == 'PRE' or cgram == 'ONO' or cgram == 'CON' or cgram.startswith('PRO') or cgram.startswith('ADJ') or cgram.startswith('ART'):
                    val = [lemma, cgram, genre, nombre]
                    #print('{}\t{}'.format(mot,val))
                    self.mot2linf[mot].append(val)
                    self.lem2lmot[lemma].append(mot)

        self.lmot = list(self.mot2linf.keys())
        sys.stderr.write('READ {} words and {} lemmas from {}\n'.format(len(self.mot2linf), len(self.lem2lmot), f))

    def get_random_txt(self, txt_curr):
        while True:
            i = random.randint(0,len(self.lmot)-1)
            txt_new = self.lmot[i]
            if txt_new != txt_curr:
                return txt_new
    
    def other_form(self, tok):
        txt_curr = tok.txt
        #print('other_form(txt_curr={})'.format(txt_curr))
        if txt_curr not in self.mot2linf:
            return '', ''
            
        linf = self.mot2linf[txt_curr]
        #print('\tlinf={}'.format(linf))
        if len(linf) != 1:
            return '', ''

        inf_curr = linf[0]
        #print('inf_curr={}'.format(inf_curr))
        lem_curr = inf_curr[0]
        tag_curr = separ.join(inf_curr[1:])
        #print('lem_curr={} tag_curr={}'.format(lem_curr,tag_curr))
        if lem_curr not in self.lem2lmot:
            return '', ''

        ltxt = self.lem2lmot[lem_curr]
        #print('ltxt={}'.format(ltxt))
        random.shuffle(ltxt)
        for txt_other in ltxt:
            if txt_other != txt_curr:
                #print('txt_other={}'.format(txt_other))
                return txt_other, tag_curr
        return '', ''
        
class Replacements():
    def __init__(self, f):
        self.tok2ltok = defaultdict(list)
        with open(f) as fd:
            n = 0
            for l in fd:
                toks = l.rstrip().split(' ')
                assert len(toks) > 0, 'bad replacement input: {}'.format(l)
                for i in range(len(toks)):
                    for j in range(i+1,len(toks)):
                        n += 2
                        self.tok2ltok[tok1].append(tok2)
                        self.tok2ltok[tok2].append(tok1)
        sys.stderr.write('READ {} replacements for {} words\n'.format(n,len(self.tok2ltok)))
        
    def replace_by(self, txt): ### {make => do, ...}
        if txt in self.tok2ltok:
            ltxt_new = self.R[txt]
            return ltxt_new[random.randint(0,len(ltxt_new)-1)]
        return ''

class Appends():
    def __init__(self, f):
        self.ltok = []
        with open(f) as fd:
            for l in fd:
                self.ltok.append(l.rstrip())
                assert len(self.ltok[-1]) > 0, 'bad appends input: {}'.format(l)
        sys.stderr.write('READ {} words to append\n'.format(len(self.ltok)))

    def is_an_append(txt_next): #### list of words that can be appended to the previous. Ex: 'avoir du' append is 'du'
        return txt_next in self.ltok

################################################################################################
################################################################################################
################################################################################################

class Noise():
    def __init__(self, args):
        self.lexicon = Lexicon(args.lexicon_file) if args.lexicon_file is not None else None
        self.replacements = Replacements(args.replace_file) if args.replace_file is not None else None
        self.appends = Appends(args.append_file) if args.append_file is not None else None
        self.args = args
        self.counts = defaultdict(int)
        
    def add(self, toks):
        self.toks = toks #list of Tok
        self.n_attempts = 0
        self.n_changes = 0
        self.output(-1) ### prints without noise
        self.seen = defaultdict(int)
        while self.n_attempts < self.args.max_tokens+5 and self.n_changes < self.args.max_tokens:
            self.n_attempts += 1
            i = random.randint(0,len(self.toks)-1) ### may be repeated
            r = random.random() ### float between [0, 1)
            p = 0.0
            if p <= r < p+self.args.p_del: ### DELETE
                self.do_delete(i)
                continue
            p += self.args.p_del
            if p <= r < p+self.args.p_rep: ### REPLACE
                self.do_replace(i)
                continue
            p += self.args.p_rep
            if p <= r < p+self.args.p_app: ### APPEND
                self.do_append(i)
                continue
            p += self.args.p_app
            if p <= r < p+self.args.p_swa: ### SWAP
                self.do_swap(i)
                continue
            p += self.args.p_swa
            if p <= r < p+self.args.p_mer: ### MERGE
                self.do_merge(i)
                continue
            p += self.args.p_mer
            if p <= r < p+self.args.p_hyp: ### HYPHEN
                self.do_hyphen(i)
                continue
            p += self.args.p_hyp
            if p <= r < p+self.args.p_spl: ### SPLIT
                self.do_split(i)
                continue
            p += self.args.p_spl
            if p <= r < p+self.args.p_cas: ### CASE
                self.do_case(i)
                continue
            p += self.args.p_cas
            if p <= r < p+self.args.p_lex: ### LEXICON
                self.do_lexicon(i)
                continue
            p += self.args.p_lex

    def output(self, i):
        out = ' '.join([t.txt+separ+t.tag for t in self.toks])
        print(out)
        if i<0:
            return
        self.n_changes += 1
        self.counts[self.toks[i].tag] += 1
        if self.args.v:
            print('n={},i={},{}\t{}'.format(self.n_changes,i,self.toks[i].tag,out))

    def stats(self):
        sys.stderr.write('Vocab of {} tags\n'.format(len(self.counts)))
        for k,v in sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True):
            sys.stderr.write('{}\t{}\n'.format(k,v))
                

    def do_delete(self, i): ### NOISY: 'i want to|DELETE to go' CORRECT: '0:i 1:want 2:to 3:go'
        if self.seen['delete']:
            return
        self.toks.insert(i, Tok(self.toks[i].txt, False, tag="DELETE") ) ## after insert(2)
        self.output(i)
        self.seen['delete'] += 1
        
    def do_replace(self, i): ### NOISY: 'i can|REPLACE_want to go' CORRECT: 'i want to go'
        if self.seen['replace']:
            return
        if self.replacements is None:
            return
        if not self.toks[i].original:
            return
        txt_old = self.toks[i].txt
        txt_new = self.replacements.replace_by(txt_old) ### want => can
        if not txt_new:
            return
        ### replace txt, tag of element i-1 avec REPLACE_tok[i]
        self.toks[i].modify(txt_new, tag='REPLACE_'+txt_old)
        self.output(i)
        self.seen['replace'] += 1
            
    def do_append(self, i): ### NOISY: 'i want|APPEND_to go' CORRECT: 'i want to go'
        ### append_by(to) => True (delete to)
        if self.seen['append']:
            return
        if self.appends is None:
            return
        if i == len(self.toks) - 1:
            return
        if not self.toks[i].original:
            return
        if not self.appends.is_an_append(self.toks[i+1].txt): #toks[i+1] is 'to'
            return
        ### remove element i+1 replace tag of element i avec APPEND_tok[i+1]
        self.toks[i].modify(self.toks[i].txt, tag='APPEND_'+self.toks[i+1].txt)
        self.toks.pop(i+1)
        self.output(i)
        self.seen['append'] += 1
        
    def do_swap(self, i): ### NOISY: 'i to|SWAP want go' CORRECT: 'i want to go'
        ### swap with next 
        if self.seen['swap']:
            return
        if i == len(self.toks) - 1:
            return
        if not self.toks[i].original or not self.toks[i+1].original:
            return
        if not self.toks[i].txt.isalpha() or not self.toks[i+1].txt.isalpha():
            return
        txt_curr = self.toks[i].txt
        txt_next = self.toks[i+1].txt
        self.toks[i].modify(txt_next,tag='SWAP')
        self.toks[i+1].modify(txt_curr,tag=keep)
        self.output(i)
        self.seen['swap'] += 1
        
    def do_merge(self, i): ### NOISY: 'i wa|MERGE nt to go' CORRECT: 'i want to go'
        ### merge two tokens
        if self.seen['merge']:
            return
        if not self.toks[i].original:
            return
        if not self.toks[i].txt.isalpha() or len(self.toks[i].txt) < 2:
            return
        k = random.randint(1,len(self.toks[i].txt)-1)
        ls = self.toks[i].txt[:k]
        rs = self.toks[i].txt[k:]
        self.toks[i].modify(''.join(rs), tag=keep)
        self.toks.insert(i,Tok(''.join(ls), False, tag='MERGE'))
        self.output(i)
        self.seen['merge'] += 1
        
    def do_hyphen(self, i): ### NOISY: 'work in depth' CORRECT: 'work in-depth'
        ### merge with an hyphen
        if self.seen['hyphen']:
            return
        if not self.toks[i].original:
            return
        p = self.toks[i].txt.find('-',1,len(self.toks[i].txt)-1)
        if p == -1:
            return
        first = self.toks[i].txt[:p]
        second = self.toks[i].txt[p+1:]
        self.toks[i].modify(second, tag=keep)
        self.toks.insert(i,Tok(first, False, tag='HYPHEN'))
        self.output(i)
        self.seen['hyphen'] += 1
        
    def do_split(self, i): ### NOISY: 'i want-to|SPLIT go' CORRECT: 'i want to go'
        ### split an hyphen
        if self.seen['split']:
            return
        if i == len(self.toks) - 1:
            return
        if not self.toks[i].original or not self.toks[i+1].original:
            return
        if not self.toks[i].txt.isalpha() or not self.toks[i+1].txt.isalpha():
            return
        self.toks[i].modify(self.toks[i].txt+'-'+self.toks[i+1].txt, tag='SPLIT')
        self.toks.pop(i+1)
        self.output(i)
        self.seen['split'] += 1
                
    def do_case(self, i): ### NOISY: 'i|CASE want to go' CORRECT: 'I want to go'
        ### change the case of the first char
        if self.seen['case']:
            return
        if not self.toks[i].original:
            return
        if not self.toks[i].txt.isalpha():
            return
        first = self.toks[i].txt[0]
        rest = self.toks[i].txt[1:]
        first = first.upper() if first.islower() else first.lower()
        self.toks[i].modify(first + rest, tag='CASE')
        self.output(i)
        self.seen['case'] += 1
        
    def do_lexicon(self, i): ### NOISY: 'i wants|VER:ind:pre:1s to go' CORRECT: 'i want to go'
        if self.seen['lexicon']:
            return
        if self.lexicon is None:
            return
        if not self.toks[i].original:
            return
        if not self.toks[i].txt.isalpha():
            return
        txt_curr_other, tag_curr = self.lexicon.other_form(self.toks[i])
        if not txt_curr_other:
            return
        self.toks[i].modify(txt_curr_other,tag=tag_curr)
        self.output(i)
        self.seen['lexicon'] += 1

        
