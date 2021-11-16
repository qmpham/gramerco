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


def g_transform(tok, tag, lexicon):
    if tok in lexicon.mot2linf:
        if tok in lexicon.mot2llempos:
            lem = lexicon.mot2llempos[tok]
            if lem in lexicon.lem2lmot:
                mots = lexicon.lem2lmot[lem]
                for mot in mots:
                    if tag in lexicon.mot2linf[mot]:
                        return mot
    return tok


def apply_tags(sentence: str, tags: torch.LongTensor, tokenizer: WordTokenizer, tagger: TagEncoder, lexicon: Lexicon):

    assert len(sentence) == len(tags)

    toks = tokenizer.tokenize(sentence)
    new_toks = list()
    i = 0
    while i < len(toks):
        tag = tagger.id_to_tag[tags[i].item()]
        if tag == '·': # keep
            new_toks.append(toks[i])
        elif "$APPEND_" in tag:
            new_toks.append(toks[i])
            new_toks.append(tag[8:])
        elif tag == "$DELETE" or tag == "$COPY":
            pass
        elif tag == "$SWAP":
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
            new_toks.append(toks[i] + toks[i+1])
            i += 1
        elif tag == "$HYPHEN":
            new_toks.append(toks[i] + '-' + toks[i+1])
            i += 1
        elif "$ART" == tag[:4] or "$PRO" == tag[:4] or "$PRE" == tag[:4] or "$ADV" == tag[:4]:
            new_toks.append(tag.split('_')[-1])
        elif tag.startswith("$TRANSFORM_"):
            new_toks.append(g_transform(toks[i], tag.split('_')[-1], lexicon))
        else:
            raise ValueError("Tag not recognized :" + tag)

        i += 1

    return sentence


def infer(args):
    tagger = TagEncoder()
    model = torch.load(args.save) if os.isfile(args.save) else GecBertModel(len(tagger))
    model.eval()
    tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
    word_tokenizer = WordTokenizer(FlaubertTokenizer)
    lexicon = Lexicon(args.lex)

    txt = args.text.split('\n')

    for i in range(len(txt) // args.batch_size + 1):
        batch_txt = txt[args.batch_size * i: min(args.batch_size * (i + 1), len(batch_txt))]
        for j in range(args.num_iter):
            toks = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                out = model(**toks) # tag_out, attention_mask
                for k, t in enumerate(batch_txt):
                    batch_txt[k] = apply_tags(
                                        t,
                                        out["tag_out"][k].argmax(-1)[out["attention_mask"][k].bool()],
                                        word_tokenizer,
                                        tagger
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

    parser.add_argument('text', help="Input file/s")
    ### optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-log', default="info", help='logging level')
    parser.add_argument('-lang', '--language', default="fr", help='language of the data')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for the training')
    parser.add_argument('--save', help='model save directory')
    parser.add_argument('--lex', default='../resources/Lexique383.tsv', help='path to lexicon table')

    args = parser.parse_args()

    create_logger("stderr", args.log)

    train(args)
