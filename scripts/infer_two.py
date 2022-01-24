from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
from data.gramerco_dataset import GramercoDataset
from model_gec.gec_bert import GecBertVocModel
from tag_encoder import TagEncoder2
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
        tags,
        tokenizer: WordTokenizer,
        tagger: TagEncoder2,
        lexicon: Lexicon,
        args,
):

    toks = tokenizer.tokenize(sentence.rstrip('\n'), max_length=510)
    # logging.info("-----------------------")
    # logging.info(sentence)
    # logging.info([tagger.id_to_tag(tag.item()) for tag in tags])
    # logging.info(len(toks))
    # logging.info(len(tags))

    if len(toks) != len(tags):
        logging.debug(len(toks))
        logging.debug(len(tags))
        toks, tags = toks[:len(tags)], tags[:len(toks)]

    if args.out_tags:
        with open(args.out_tags, 'a') as f:
            f.write(' '.join(['|'.join(e) for e in zip(toks, tags)]) + '\n')

    assert len(toks) == len(tags)

    new_toks = list()
    i = 0
    while i < len(toks):
        tag = tags[i]
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
        elif tag == "$CASE:FIRST":
            if toks[i][0].isupper():
                new_toks.append(toks[i][0].lower() + toks[i][1:])
            elif toks[i][0].islower():
                new_toks.append(toks[i][0].upper() + toks[i][1:])
        elif tag == "$CASE:UPPER":
            new_toks.append(toks[i].upper())
        elif tag == "$CASE:LOWER":
            new_toks.append(toks[i].lower())
        elif tag == "$HYPHEN:SPLIT":
            for t in toks[i].split('-'):
                new_toks.append(t)
        elif tag == "$MERGE":
            if i != len(toks) - 1:
                new_toks.append(toks[i] + toks[i + 1])
            else:
                new_toks.append(toks[i])
            i += 1
        elif tag == "$HYPHEN:MERGE":
            if i != len(toks) - 1:
                new_toks.append(toks[i] + '-' + toks[i + 1])
            else:
                new_toks.append(toks[i])
            i += 1
        elif tag.startswith("$REPLACE"):
            new_toks.append(tag.split('_')[-1])
        elif tag.startswith("$INFLECT_"):
            new_toks.append(g_transform(toks[i], tag.split('_')[-1], lexicon))
        elif tag.startswith("$SPLIT"):
            w = tag.split('_')[-1]
            if toks[i].startswith(w):
                new_toks.append(w)
                new_toks.append(toks[i][len(w):])
            else:
                new_toks.append(toks[i])
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
    tagger = TagEncoder2(
        path_to_lex=args.lex,
        path_to_voc=args.voc,
    )

    path_to_model = os.path.join(
        args.save_path,
        args.model_id,
        "model_best.pt"
    )
    model = GecBertVocModel(
        len(tagger),
        len(tagger.worder),
        tokenizer=tokenizer,
        tagger=tagger,
        mid=args.model_id,
    )

    device = "cuda:" + str(args.gpu_id) \
        if args.gpu and torch.cuda.is_available() else "cpu"
    if os.path.isfile(path_to_model):
        state_dict = torch.load(path_to_model, map_location=torch.device(device))
        model.load_state_dict(state_dict["model_state_dict"])

    else:
        logging.info("Model not found at: " + path_to_model)
        return
    model.eval()

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

    if args.out_tags:
        with open(args.out_tags, 'w') as f:
            f.write('')
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
            del toks["token_type_ids"]
            xi = toks["input_ids"][0][
                toks["attention_mask"][0].bool()
            ]
            # logging.info("noise >>> " + " ".join(map(tokenizer._convert_id_to_token, xi.cpu().numpy())))
            # logging.info(toks["input_ids"].shape)
            with torch.no_grad():
                out = model(**toks)  # tag_out, attention_mask
                for k, t in enumerate(batch_txt):
                    tag_ids = out["tag_out"][k].argmax(-1)[out["attention_mask"][k].bool()]
                    voc_ids = out["voc_out"][k].argmax(-1)[out["attention_mask"][k].bool()]

                    tags = [tagger._id_to_tag[i.item()] for i in tag_ids]
                    vocs = [tagger.worder.id_to_word[i.item()] for i in voc_ids]

                    for i, tag in enumerate(tags):
                        if tagger.is_radical_word_tag(tag):
                            tags[i] = tag + '_' + vocs[i]
                    logging.info('-' * 50)
                    logging.info(batch_txt[k].rstrip('\n'))
                    logging.info(' '.join(tags))

                    batch_txt[k] = apply_tags(
                        t,
                        tags,
                        word_tokenizer,
                        tagger,
                        lexicon,
                        args,
                    )
                    # logging.info(str(dec.long().cpu().numpy()))
                    # yy = out["tag_out"][k][out["attention_mask"][k].bool()]
                    # yy = torch.softmax(yy, -1)
                    # jj = yy.topk(3, dim=-1).indices.cpu()
                    # ii = torch.arange(
                    #     jj.size(0)).unsqueeze(-1).expand(jj.shape)
                    # logging.info(jj)
                    # logging.info(yy[ii, jj])
                    # for topk in range(3):
                    #     logging.info(
                    #         " ".join(
                    #             tagger.id_to_tag(
                    #                 tid.item()) for tid in jj[:, topk]))
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
        '--voc',
        required=True,
        help="Path to appendable data."
    )
    parser.add_argument(
        '--tokenizer',
        default="flaubert/flaubert_base_cased",
        help='model save directory'
    )
    parser.add_argument(
        '--out-tags',
        default=None,
        help='file for the tagged output words'
    )

    args = parser.parse_args()

    create_logger("stderr", args.log)

    infer(args)
