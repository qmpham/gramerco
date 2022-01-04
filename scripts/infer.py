from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
from data.gramerco_dataset import GramercoDataset
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


def apply_tags(
        sentence: str,
        tags: torch.LongTensor,
        tokenizer: WordTokenizer,
        tagger: TagEncoder,
        lexicon: Lexicon):

    toks = tokenizer.tokenize(sentence, max_length=510)
    # logging.info("-----------------------")
    # logging.info(sentence)
    # logging.info([tagger.id_to_tag(tag.item()) for tag in tags])
    # logging.info(len(toks))
    # logging.info(len(tags))

    if len(toks) != len(tags):
        logging.debug(len(toks))
        logging.debug(len(tags))
        toks, tags = toks[:len(tags)], tags[:len(toks)]

    assert len(toks) == len(tags)

    new_toks = list()
    i = 0
    while i < len(toks):
        tag = tagger.id_to_tag(tags[i].item())
        if tag == '·':  # keep
            new_toks.append(toks[i])
        elif "$APPEND_" in tag:
            new_toks.append(toks[i])
            new_toks.append(tag[8:])
        elif tag == "$DELETE" or tag == "$COPY":
            pass
        elif tag == "$SWAP":
            if i != len(toks) - 1:
                new_toks.append(toks[i + 1])
            new_toks.append(toks[i])
            i = i + 1
        elif tag == "$CASE":
            if toks[i][0].isupper():
                new_toks.append(toks[i][0].lower() + toks[i][1:])
            elif toks[i][0].islower():
                new_toks.append(toks[i][0].upper() + toks[i][1:])
        elif tag == "$SPLIT":
            for t in toks[i].split('-'):
                new_toks.append(t)
        elif tag == "$MERGE":
            if i != len(toks) - 1:
                new_toks.append(toks[i] + toks[i + 1])
            else:
                new_toks.append(toks[i])
            i += 1
        elif tag == "$HYPHEN":
            if i != len(toks) - 1:
                new_toks.append(toks[i] + '-' + toks[i + 1])
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


def infer(args):

    tokenizer = FlaubertTokenizer.from_pretrained(
        args.tokenizer
    )
    word_tokenizer = WordTokenizer(FlaubertTokenizer)
    lexicon = Lexicon(args.lex)
    tagger = TagEncoder(
        path_to_lex=args.lex,
        path_to_app=args.app,
    )

    path_to_model = os.path.join(
        args.save_path,
        args.model_id,
        "model_best.pt"
    )
    model = model = GecBertModel(
        len(tagger),
        tagger=tagger,
        tokenizer=tokenizer,
        mid=args.model_id,
    )
    if os.path.isfile(path_to_model):
        state_dict = torch.load(path_to_model)
        if isinstance(state_dict, GecBertModel):
            model = state_dict
        else:
            model.load_state_dict(state_dict["model_state_dict"])

    else:
        logging.info("Model not found at: " + path_to_model)
        return
    model.eval()
    device = "cuda:" + str(args.gpu_id) \
        if args.gpu and torch.cuda.is_available() else "cpu"
    logging.info(torch.cuda.device_count())
    logging.info("device = " + device)
    model.to(device)

    if args.text:
        txt = args.text.split('\n')
    elif args.file:
        with open(args.file, 'r') as f:
            txt = f.readlines()
    else:
        raise ValueError("No input argument. try --text or --file")
    # logging.info("-----------------------")
    # logging.info(txt)
    # logging.info([tagger.id_to_tag(tag.item()) for tag in tags])
    # logging.info(len(toks))
    # logging.info(len(tags))

    for i in range(len(txt) // args.batch_size + 1):
        if i > 0:
            print()
        batch_txt = txt[args.batch_size *
                        i: min(args.batch_size * (i + 1), len(txt))]
        for j in range(args.num_iter):
            toks = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=510
            ).to(device)
            logging.info(toks["input_ids"].shape)
            with torch.no_grad():
                out = model(**toks)  # tag_out, attention_mask
                for k, t in enumerate(batch_txt):
                    batch_txt[k] = apply_tags(
                        t,
                        out["tag_out"][k].argmax(-1)[out["attention_mask"][k].bool()].cpu(),
                        word_tokenizer,
                        tagger,
                        lexicon
                    )
                    yy = out["tag_out"][k][out["attention_mask"][k].bool()]
                    yy = torch.softmax(yy, -1)
                    jj = yy.topk(3, dim=-1).indices.cpu()
                    ii = torch.arange(
                        jj.size(0)).unsqueeze(-1).expand(jj.shape)
                    logging.info(jj)
                    logging.info(yy[ii, jj])
                    for topk in range(3):
                        logging.info(
                            " ".join(
                                tagger.id_to_tag(
                                    tid.item()) for tid in jj[:, topk]))
        print('\n'.join(batch_txt))


def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(
            format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s',
            datefmt='%Y-%m-%d_%H:%M:%S',
            level=numeric_level)
    else:
        logging.basicConfig(
            filename=logfile,
            format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s',
            datefmt='%Y-%m-%d_%H:%M:%S',
            level=numeric_level)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--text', default=None, help="Input text")
    parser.add_argument('--file', default=None, help="Input file")
    # optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('--log', default="info", help='logging level')
    parser.add_argument(
        '--gpu',
        action='store_true',
        help="GPU usage activation."
    )
    parser.add_argument(
        '--gpu-id',
        default=0,
        type=int,
        help="GPU id, generally 0 or 1."
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='batch size for eval'
    )
    parser.add_argument(
        '--num-iter',
        type=int,
        default=1,
        help='num iteration loops to edit'
    )
    parser.add_argument(
        '--save-path',
        required=True,
        help='model save directory'
    )
    parser.add_argument(
        '--model-id',
        required=True,
        help="Model id (folder name)"
    )
    parser.add_argument(
        '--lex',
        required=True,
        help='path to lexicon table.'
    )
    parser.add_argument(
        '--app',
        required=True,
        help="Path to appendable data."
    )
    parser.add_argument(
        '--tokenizer',
        default="flaubert/flaubert_base_cased",
        help='model save directory'
    )

    args = parser.parse_args()

    create_logger("stderr", args.log)

    infer(args)
