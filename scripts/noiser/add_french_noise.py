#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import random
import logging
from collections import defaultdict
from tqdm import tqdm

pwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(pwd))
sys.path.append(os.path.dirname(os.path.abspath(pwd)))
from tokenizer import WordTokenizer
from transformers import FlaubertTokenizer


separ = "￨"
keep = "·"
space = "~"


def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == "stderr":
        logging.basicConfig(
            format="[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S",
            level=numeric_level,
        )
    else:
        logging.basicConfig(
            filename=logfile,
            format="[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S",
            level=numeric_level,
        )


def fixCase(other, curr):
    if curr[0].isupper() and other[0].islower():
        return other[0].upper() + other[1:]
    return other


def output(toks, tags, output_file):
    out = []
    for i in range(len(toks)):
        out.append(toks[i] + separ + tags[i])
    # print(" ".join(out))
    output_file.write(" ".join(out))


def do_lexicon(toks, tags, lex_rep):
    random.shuffle(lex_rep)
    for lex in lex_rep:  # lex is: $VER￨-￨-￨ind￨pre￨3s:chutes￨chute
        sp = lex.split(":")
        if len(sp) != 2:
            continue
        tag, others = lex.split(":")[:2]
        others = others.split(separ)
        curr = others.pop()
        logging.debug(
            "tag={} curr={} others={} n_found={} toks={}".format(
                tag, curr, others, toks.count(curr), toks
            )
        )
        if toks.count(curr) != 1:
            continue
        random.shuffle(others)
        other = others[0]
        idx = toks.index(curr)
        if tags[idx] != keep:
            continue
        tags[idx] = "$TRANSFORM_" + tag
        toks[idx] = fixCase(other, curr)
        logging.debug("do_lexicon: {} => {} tag={}".format(curr, toks[idx], tags[idx]))
        return tags[idx]
    return ""


def do_delete(toks, tags, dic):
    ### insert a random word in a random position (consider inserting a copy of the previous)
    if len(dic["txt"]) == 0:
        return ""
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        txt_new = random.choices(dic["txt"], weights=dic["frq"], k=1)[0]
        if toks[idx] == txt_new:
            continue
        tag_new = "$DELETE"
        toks.insert(idx, txt_new)
        tags.insert(idx, tag_new)
        return tag_new
    return ""


def do_copy(toks, tags):
    ### copy a word and tag the first as COPY
    if len(dic["txt"]) == 0:
        return ""
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if not toks[idx].isalpha():
            continue
        toks.insert(idx, toks[idx])
        tags.insert(idx, "$COPY")
        return tags[idx]
    return ""


def do_replace(toks, tags, rep):
    if len(rep["mot2pos"]) == 0:
        return ""
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if toks[idx] not in rep["mot2pos"]:
            continue
        pos = rep["mot2pos"][toks[idx]]
        if len(pos) > 1:
            continue
        pos = list(pos)[0]
        others = list(rep["pos2mot"][pos])
        random.shuffle(others)
        other = list(others)[0]
        txt = toks[idx]
        toks[idx] = other
        tags[idx] = "$" + pos + "_" + txt
        return tags[idx]
    return ""


def do_append(toks, tags, app):
    ### remove the next word if word in append, tag the previous as APPEND_word
    idxs = list(range(len(toks) - 1))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep or tags[idx + 1] != keep:
            continue
        if not toks[idx].isalpha():
            continue
        if toks[idx + 1] not in app:
            continue
        tags[idx] = "$APPEND_" + toks[idx + 1]
        toks.pop(idx + 1)
        tags.pop(idx + 1)
        return tags[idx]
    return ""


def do_swap(toks, tags):
    # replace i and i+1
    idxs = list(range(len(toks) - 1))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep or tags[idx + 1] != keep:
            continue
        curr_txt = toks[idx]
        curr_tag = tags[idx]
        toks[idx] = toks[idx + 1]
        tags[idx] = "$SWAP"
        toks[idx + 1] = curr_txt
        tags[idx + 1] = curr_tag
        return tags[idx]
    return ""


def do_merge(toks, tags):
    ### split toks[idx] in two tokens and tag the first to MERGE
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if len(toks[idx]) < 2:
            continue
        if not toks[idx].isalpha():
            continue
        k = random.randint(1, len(toks[idx]) - 1)
        ls = toks[idx][:k]
        rs = toks[idx][k:]
        toks[idx] = "".join(rs)
        tags[idx] = keep
        toks.insert(idx, "".join(ls))
        tags.insert(idx, "$MERGE")
        return tags[idx]
    return ""


def do_case(toks, tags):
    ### change the case of first char in token
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if len(toks[idx]) < 2:
            continue
        if not toks[idx].isalpha():
            continue
        first = toks[idx][0]
        rest = toks[idx][1:]
        first = first.lower() if first.isupper() else first.upper()
        toks[idx] = first + rest
        tags[idx] = "$CASE"
        return tags[idx]
    return ""


def do_split(toks, tags):
    ### take two consecutive words and join them with an hyphen, tag it as SPLIT
    idxs = list(range(len(toks) - 1))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep or tags[idx + 1] != keep:
            continue
        if not toks[idx].isalpha() or not toks[idx + 1].isalpha():
            continue
        toks[idx] = toks[idx] + "-" + toks[idx + 1]
        tags[idx] = "$SPLIT"
        toks.pop(idx + 1)
        tags.pop(idx + 1)
        return tags[idx]
    return ""


def do_hyphen(toks, tags):
    ### take a word with an hyphen and split then in two, tag the first as HYPHEN
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if toks[idx].count("-") != 1:
            continue
        p = toks[idx].find("-", 1, len(toks[idx]) - 1)
        if p == -1:
            continue
        first = toks[idx][:p]
        second = toks[idx][p + 1 :]
        toks[idx] = second
        toks.insert(idx, first)
        tags.insert(idx, "$HYPHEN")
        return tags[idx]
    return ""


def noise_line(toks, lex, dic, rep, app, seen, args, out):
    n_attempts = 0
    n_changes = 0
    tags = [keep for t in toks]
    output(toks, tags, out)  ### prints without noise
    while n_attempts < args.max_tokens * 2 and n_changes < args.max_tokens:
        n_attempts += 1
        r = random.random()  ### float between [0, 1)
        p = 0.0

        if p <= r < p + args.p_lex:  ### LEXICON #######################
            tag = do_lexicon(toks, tags, lex)  # tag may be '' if not replacement
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_lex

        if p <= r < p + args.p_rep:  ### REPLACE #######################
            tag = do_replace(toks, tags, rep)
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_rep

        if p <= r < p + args.p_app:  ### APPEND #######################
            tag = do_append(toks, tags, app)
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_app

        if p <= r < p + args.p_del:  ### DELETE #######################
            tag = do_delete(toks, tags, dic)
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_del

        if p <= r < p + args.p_cop:  ### COPY #######################
            tag = do_copy(toks, tags)
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_del

        if p <= r < p + args.p_swa:  ### SWAP #######################
            tag = do_swap(toks, tags)
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_swa

        if p <= r < p + args.p_mer:  ### MERGE #######################
            tag = do_merge(toks, tags)
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_mer

        if p <= r < p + args.p_hyp:  ### HYPHEN #######################
            tag = do_hyphen(toks, tags)
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_hyp

        if p <= r < p + args.p_spl:  ### SPLIT #######################
            tag = do_split(toks, tags)
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_spl

        if p <= r < p + args.p_cas:  ### CASE #######################
            tag = do_case(toks, tags)
            if tag:
                output(toks, tags, out)
                seen[tag] += 1
                n_changes += 1
            continue
        p += args.p_cas


def read_dic(f):
    txt = []
    frq = []
    if not f:
        return {"txt": txt, "frq": frq}
    with open(f, "r") as fd:
        for l in fd:
            w, f = l.rstrip().split()
            txt.append(w)
            frq.append(float(f))
    return {"txt": txt, "frq": frq}


def read_rep(f):
    mot2pos = defaultdict(set)
    pos2mot = defaultdict(set)
    if not f:
        return {"mot2pos": mot2pos, "pos2mot": pos2mot}
    with open(f, "r") as fd:
        for l in fd:
            toks = l.rstrip().split("\t")
            mot, cgram = toks[0].replace(" ", space), toks[3]
            if cgram.startswith("ART"):
                mot2pos[mot].add("ART")
                pos2mot["ART"].add(mot)
            if cgram.startswith("PRO"):
                mot2pos[mot].add("ART")
                pos2mot["PRO"].add(mot)
            elif cgram == "PRE" or cgram == "ADV":
                mot2pos[mot].add(cgram)
                pos2mot[cgram].add(mot)
    return {"mot2pos": mot2pos, "pos2mot": pos2mot}


def read_app(f):
    mots = list()
    if not f:
        return mots
    with open(f, "r", encoding="ISO-8859-1") as fd:
        for l in fd:
            mots.append(l.rstrip())
    return mots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="Input file/s", nargs="+")
    parser.add_argument(
        "--out", required=True, help="Destination output file.",
    )
    ### optional
    parser.add_argument("-v", action="store_true")
    parser.add_argument(
        "--force-replace",
        action="store_true",
        help="File with list of words and their frequencies",
    )
    parser.add_argument(
        "-max",
        "--max_tokens",
        type=int,
        default=3,
        help="Attempts to inject noise in this number of tokens per sentence (def 3)",
    )
    parser.add_argument(
        "-dic",
        "--dictionary_file",
        help="File with list of words and their frequencies",
    )
    parser.add_argument("-rep", "--replace_file", help="Path of Lexique383.tsv")
    parser.add_argument("-app", "--append_file", help="File with words to be appended")
    parser.add_argument(
        "--p_lex",
        type=float,
        default=0.1,
        help="Prob of transforming a token using morph features (def 0.1) [to use with -lex]",
    )
    parser.add_argument(
        "--p_app",
        type=float,
        default=0.05,
        help="Prob of appending one token (def 0.05) [to use with -app]",
    )
    parser.add_argument(
        "--p_rep",
        type=float,
        default=0.05,
        help="Prob of replacing one token (def 0.05) [to use with -rep]",
    )
    parser.add_argument(
        "--p_del",
        type=float,
        default=0.01,
        help="Prob of deleting one token (def 0.01)",
    )
    parser.add_argument(
        "--p_cop", type=float, default=0.01, help="Prob of copying one token (def 0.01)"
    )
    parser.add_argument(
        "--p_mer",
        type=float,
        default=0.01,
        help="Prob of merging two tokens (def 0.01)",
    )
    parser.add_argument(
        "--p_spl",
        type=float,
        default=0.01,
        help="Prob of splitting one token (def 0.01)",
    )
    parser.add_argument(
        "--p_hyp",
        type=float,
        default=0.01,
        help="Prob of joining two tokens with an hyphen (def 0.01)",
    )
    parser.add_argument(
        "--p_swa",
        type=float,
        default=0.01,
        help="Prob of swapping two tokens (def 0.01)",
    )
    parser.add_argument(
        "--p_cas",
        type=float,
        default=0.01,
        help="Prob of changing the first char case (def 0.01)",
    )
    parser.add_argument(
        "-log",
        default="info",
        help="Logging level [debug, info, warning, critical, error] (info)",
    )
    args = parser.parse_args()
    create_logger("stderr", args.log)
    t = WordTokenizer(FlaubertTokenizer)
    dic = read_dic(args.dictionary_file)
    rep = read_rep(args.replace_file)
    app = read_app(args.append_file)
    nsents = 0
    ntokens = 0
    seen = defaultdict(int)
    if not args.force_replace:
        f_out = args.out
        logging.info(f_out)
        logging.info(os.path.isfile(f_out))
        logging.info(os.path.getsize(f_out))
        if os.path.isfile(f_out) and os.path.getsize(f_out) > 10000:
            sys.exit()
    output_file = open(args.out, "w")
    for f in args.files:

        with open(f, "r", encoding="ISO-8859-1") as fd:
            lines = []
            for l in fd:
                lines.append(l.rstrip())
        sys.stderr.write("READ {} sentences from {}\n".format(len(lines), f))
        pbar = tqdm(lines)
        pbar.set_description("Processing {}".format(os.path.basename(f)))
        for line in pbar:
            line = line.split("\t")
            toks = t.tokenize(line.pop(0))
            noise_line(toks, line, dic, rep, app, seen, args, output_file)
            # print()
            output_file.write("\n")
            nsents += 1
            ntokens += len(toks)
        sys.stderr.write("Found {} sentences\n".format(nsents))

        logging.debug("Vocab of {} tags".format(len(seen)))
        for k, v in sorted(seen.items(), key=lambda kv: kv[1], reverse=True):
            logging.debug("{}\t{}".format(v, k))
    output_file.close()
