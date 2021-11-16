from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
from data.data import GramercoDataset
from model_gec.gec_bert import GecBertModel
from tag_encoder import TagEncoder
from tokenizer import WordTokenizer
import logging
import matplotlib.pyplot as plt
from noiser.Noise import Lexicon
import os
import sys
import re


separ = '￨'


def g_transform(tok, tag, lexicon):
    if tok in lexicon.mot2linf:
        if tok in lexicon.mot2llempos:
            lems = lexicon.mot2llempos[tok]
            for lem in lems:
                lem = lem.split(separ)[0]
                if lem in lexicon.lem2lmot:
                    mots = lexicon.lem2lmot[lem]
                    for mot in mots:
                        if tag in lexicon.mot2linf[mot]:
                            return mot
    return tok


def apply_tags(sentence: str, tags: torch.LongTensor, tokenizer: WordTokenizer, tagger: TagEncoder, lexicon: Lexicon):

    toks = tokenizer.tokenize(sentence, max_length=510)
    # logging.info("-----------------------")
    # logging.info(sentence)
    # logging.info([tagger.id_to_tag(tag.item()) for tag in tags])
    # logging.info(len(toks))
    # logging.info(len(tags))

    if len(toks) != len(tags):
        logging.info(len(toks))
        logging.info(len(tags))
        toks, tags = toks[:len(tags)], tags[:len(toks)]

    assert len(toks) == len(tags)

    new_toks = list()
    i = 0
    while i < len(toks):
        tag = tagger.id_to_tag(tags[i].item())
        if tag == '·': # keep
            new_toks.append(toks[i])
        elif "$APPEND_" in tag:
            new_toks.append(toks[i])
            new_toks.append(tag[8:])
        elif tag == "$DELETE" or tag == "$COPY":
            pass
        elif tag == "$SWAP":
            if i != len(toks) - 1:
                new_toks.append(toks[i+1])
            new_toks.append(toks[i])
            i = i + 1
        elif tag == "$CASE":
            if toks[i][0].isupper():
                new_toks.append(toks[i][0].lower() + toks[i][1:])
            elif toks[i][0].islower():
                new_toks.append(toks[i][0].upper() + toks[i][1:])
        elif tag == "$SPLIT":
            for t in  toks[i].split('-'):
                new_toks.append(t)
        elif tag == "$MERGE":
            if i != len(toks) - 1:
                new_toks.append(toks[i] + toks[i+1])
            else:
                new_toks.append(toks[i])
            i += 1
        elif tag == "$HYPHEN":
            if i != len(toks) - 1:
                new_toks.append(toks[i] + '-' + toks[i+1])
            else:
                new_toks.append(toks[i])
            i += 1
        elif "$ART" == tag[:4] or "$PRO" == tag[:4] or "$PRE" == tag[:4] or "$ADV" == tag[:4]:
            new_toks.append(tag.split('_')[-1])
        elif tag.startswith("$TRANSFORM_"):
            new_toks.append(g_transform(toks[i], tag.split('_')[-1], lexicon))
        else:
            raise ValueError("Tag not recognized :" + tag)

        i += 1


    # logging.info(len(new_toks))
    new_sentence = ' '.join(new_toks[:510])
    new_sentence = re.sub("' ", "'", new_sentence)

    # logging.info(new_sentence)
    # logging.info("**********************")

    return new_sentence


def infer(args, text=""):
    tagger = TagEncoder()
    model = torch.load(args.load) if os.path.isfile(args.load) else GecBertModel(len(tagger))
    model.eval()
    tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
    word_tokenizer = WordTokenizer(FlaubertTokenizer)
    lexicon = Lexicon(args.lex)

    logging.info(text)

    txt = text.split('\n')
    # logging.info("-----------------------")
    # logging.info(txt)
    # logging.info([tagger.id_to_tag(tag.item()) for tag in tags])
    # logging.info(len(toks))
    # logging.info(len(tags))

    for i in range(len(txt) // args.batch_size + 1):
        batch_txt = txt[args.batch_size * i: min(args.batch_size * (i + 1), len(txt))]
        for j in range(args.num_iter):
            toks = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=510
            )
            logging.info(toks["input_ids"].shape)
            with torch.no_grad():
                out = model(**toks) # tag_out, attention_mask
                for k, t in enumerate(batch_txt):
                    batch_txt[k] = apply_tags(
                                        t,
                                        out["tag_out"][k].argmax(-1)[out["attention_mask"][k].bool()],
                                        word_tokenizer,
                                        tagger,
                                        lexicon
                                    )
        print('\n'.join(batch_txt))


def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
    else:
        logging.basicConfig(filename=logfile, format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--text', default=None, help="Input file")
    ### optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-log', default="info", help='logging level')
    parser.add_argument('-lang', '--language', default="fr", help='language of the data')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for eval')
    parser.add_argument('--num-iter', type=int, default=4, help='num iteration loops to edit')
    parser.add_argument('--load', default='', help='model save directory')
    parser.add_argument('--lex', default='../resources/Lexique383.tsv', help='path to lexicon table')

    args = parser.parse_args()

    create_logger("stderr", args.log)

    infer(args, text=sys.stdin.read() if args.text is None else args.text)
