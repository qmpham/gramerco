from collections import defaultdict
import torch
from noiser.Noise import Spacy
from transformers import FlaubertTokenizer
import os
import sys
import logging
pwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(pwd))
sys.path.append(os.path.dirname(os.path.abspath(pwd)))
try:
    from noiser.add_french_noise import read_rep, read_app
except BaseException:
    from .noiser.add_french_noise import read_rep, read_app

separ = '￨'



def default_keep_tok():
    return '·'

def default_empty():
    return ''

def default_keep_id():
    return 0


class TagEncoder:
    def __init__(
            self,
            path_to_lex="/home/bouthors/workspace/gramerco-repo/gramerco/resources/Lexique383.tsv",
            path_to_app="/home/bouthors/workspace/gramerco-repo/gramerco/resources/lexique.app",
    ):
        f = open(path_to_app, 'r')
        f.close()
        rep = read_rep(path_to_lex)
        app = read_app(path_to_app)

        self._id_to_tag = defaultdict(default_keep_tok)
        self._tag_to_id = defaultdict(default_keep_id)
        self._curr_cpt = 1

        self.error_type_id = {
            "DELETE": 0,
            "COPY": 1,
            "SWAP": 2,
            "MERGE": 3,
            "CASE": 4,
            "SPLIT": 5,
            "HYPHEN": 6,
            "APPEND": 7,
            "TRANSFORM": 8,
            "REPLACE": 9,
        }

        self.id_error_type = [
            "DELETE",
            "COPY",
            "SWAP",
            "MERGE",
            "CASE",
            "SPLIT",
            "HYPHEN",
            "APPEND",
            "TRANSFORM",
            "REPLACE",
        ]

        self.add_tag("$DELETE")
        self.add_tag("$COPY")
        self.add_tag("$SWAP")
        self.add_tag("$MERGE")
        self.add_tag("$CASE")
        self.add_tag("$SPLIT")
        self.add_tag("$HYPHEN")

        for pos in ["ADJ", "NOM"]:
            for genre in ["m", "f", "-"]:
                for nombre in ["s", "p", "-"]:
                    self.add_tag("$TRANSFORM_" +
                                 separ.join([pos, genre, nombre]))

        for pos in ["VER", "AUX"]:
            self.add_tag("$TRANSFORM_" + separ.join([pos, "-", "-", "inf"]))

            for genre in ["m", "f"]:
                for nombre in ["s", "p"]:
                    for tense in ["pas", "pre"]:
                        self.add_tag("$TRANSFORM_" +
                                     separ.join([pos, genre, nombre, "par", tense]))

            for tense in ["pas", "pre", "fut", "imp"]:
                for nombre in ["s", "p"]:
                    for pers in ["1", "2", "3"]:
                        self.add_tag(
                            "$TRANSFORM_" + separ.join([pos, "-", "-", "ind", tense, pers + nombre]))

            for nombre in ["s", "p"]:
                for pers in ["1", "2", "3"]:
                    self.add_tag(
                        "$TRANSFORM_" + separ.join([pos, "-", "-", "sub", "pre", pers + nombre]))

        for app_tok in app:
            self.add_tag("$APPEND_" + app_tok)

        for pos in rep["pos2mot"]:  # ART + PRO + PRE + ADV
            for tok in rep["pos2mot"][pos]:
                self.add_tag("$" + pos + "_" + tok)

    def encode_line(self, line):
        return torch.tensor(
            list(map(self.tag_to_id, line.split(" "))), dtype=torch.int64)

    def id_to_tag(self, i):
        return self._id_to_tag[i]

    def tag_to_id(self, tag):
        return self._tag_to_id[tag]

    def add_tag(self, tag):
        self._id_to_tag[self._curr_cpt] = tag
        self._tag_to_id[tag] = self._curr_cpt
        self._curr_cpt += 1

    def get_tag_category(self, tag):
        if type(tag) == int:
            # logging.debug(str(tag))
            tag = self.id_to_tag(tag)
        error_type = tag[1:].split('_')[0]
        if error_type in ["ART", "PRO", "PRE", "ADV"]:
            error_type = "REPLACE"
        # logging.debug(error_type + "   :   " + tag)
        # DELETE, COPY, SWAP, SPLIT, HYPHEN, CASE, TRANSFORM, APPEND, REPLACE
        if error_type in error_type_id:
            return error_type_id[error_type]
        return error_type_id["KEEP"]

    def size(self):
        return self._curr_cpt

    def __len__(self):
        return self.size()


class TagEncoder2(TagEncoder):

    def __init__(
            self,
            path_to_lex="/home/bouthors/workspace/gramerco-repo/gramerco/resources/Lexique383.tsv",
            path_to_app="/home/bouthors/workspace/gramerco-repo/gramerco/resources/lexique.app",
    ):

        rep = read_rep(path_to_lex)
        app = read_app(path_to_app)
        self.worder = WordEncoder(path_to_app)

        self._id_to_tag = defaultdict(default_keep_tok)
        self._tag_to_id = defaultdict(default_keep_id)
        self._curr_cpt = 1

        self.id_error_type = [
            "DELETE",
            "COPY",
            "SWAP",
            "SPLIT",
            "MERGE",
            "SPLIT_HYPHEN",
            "MERGE_HYPHEN",
            "CASE",
            "CASE_UPPER",
            "CASE_LOWER",
            "TRANSFORM",
            "APPEND",
            "REPLACE",
            "SPLIT",
        ]

        self.error_type_id = {
            key: i for i, key in enumerate(self.id_error_type)
        }

        self.add_tag("$DELETE")
        self.add_tag("$COPY")
        self.add_tag("$SWAP")
        # self.add_tag("$SPLIT")
        self.add_tag("$MERGE")
        self.add_tag("$SPLIT_HYPHEN")
        self.add_tag("$MERGE_HYPHEN")
        self.add_tag("$CASE")
        self.add_tag("$CASE_UPPER")
        self.add_tag("$CASE_LOWER")

        for pos in ["ADJ", "NOM"]:
            for genre in ["m", "f", "-"]:
                for nombre in ["s", "p", "-"]:
                    self.add_tag("$TRANSFORM_" +
                                 separ.join([pos, genre, nombre]))

        for pos in ["VER", "AUX"]:
            self.add_tag("$TRANSFORM_" + separ.join([pos, "-", "-", "inf"]))

            for genre in ["m", "f"]:
                for nombre in ["s", "p"]:
                    for tense in ["pas", "pre"]:
                        self.add_tag("$TRANSFORM_" +
                                     separ.join([pos, genre, nombre, "par", tense]))

            for tense in ["pas", "pre", "fut", "imp"]:
                for nombre in ["s", "p"]:
                    for pers in ["1", "2", "3"]:
                        self.add_tag(
                            "$TRANSFORM_" + separ.join([pos, "-", "-", "ind", tense, pers + nombre]))

            for nombre in ["s", "p"]:
                for pers in ["1", "2", "3"]:
                    self.add_tag(
                        "$TRANSFORM_" + separ.join([pos, "-", "-", "sub", "pre", pers + nombre]))

        self.add_tag("$APPEND")
        self.add_tag("$REPLACE")
        self.add_tag("$SPLIT")
        # for app_tok in app:
        #     self.add_tag("$APPEND_" + app_tok)
        #
        # for pos in rep["pos2mot"]:  # ART + PRO + PRE + ADV
        #     for tok in rep["pos2mot"][pos]:
        #         self.add_tag("$REPLACE_" + tok)

    def id_to_tag(self, i):
        if i < self.size - 3:
            self._id_to_tag[i]
        j = i - self._curr_cpt + 3
        tag = j // len(self.worder)
        word = j % len(self.worder)

        return (
            "$" +
            self.id_error_type[self._curr_cpt - tag - 1] +
            "_" +
            self.worder.id_to_word[word]
        )

    def tag_to_id(self, tag):
        if self.is_word_tag(tag):
            tags = tag.split('_')
            word = tags[-1]
            word = self.worder.word_to_id[word]
            tag = '_'.join(tags[:-1])[1:]
            cls = self.error_type_id[tag]
            cls = self._curr_cpt - cls - 1
            return self._curr_cpt - 3 + len(worder) * cls + word

        return self._tag_to_id[tag[1:]]

    def is_word_tag(tag):
        return (tag.startswith("$APPEND")
                or tag.startswith("$REPLACE")
                or (tag.startswith("$SPLIT")
                    and tag != "$SPLIT_HYPHEN"))

    def id_to_type_id(tid):
        tag = self.id_to_tag(tid)
        if self.is_word_tag(tag):
            name = '_'.join(self.id_to_tag(tid).split('_')[:-1])[1:]
            return self.error_type_id[name]
        return self.error_type_id[tag[1:]]

    def id_to_word(tid):
        tag = self.id_to_tag(tid)
        if self.is_word_tag(tag):
            return tag.split('_')[-1]


class WordEncoder:

    def __init__(
        self,
        path_to_voc="/home/bouthors/workspace/gramerco-repo/gramerco/resources/lexique.app",
    ):
    voc = read_app(path_to_voc)
    self.id_to_word = defaultdict(default_keep_id)
    self.word_to_id = default_dict(default_empty)
    self._curr_cpt = 0
    for word in voc:
        self.add_word(word)

    def add_word(self, word):
        self._id_to_word[self._curr_cpt] = word
        self._word_to_id[word] = self._curr_cpt
        self._curr_cpt += 1

    def get_tag_category(self, tag):
        if type(tag) == int:
            # logging.debug(str(tag))
            tag = self.id_to_tag(tag)
        error_type = tag[1:].split('_')[0]
        if error_type in ["ART", "PRO", "PRE", "ADV"]:
            error_type = "REPLACE"
        # logging.debug(error_type + "   :   " + tag)
        # DELETE, COPY, SWAP, SPLIT, HYPHEN, CASE, TRANSFORM, APPEND, REPLACE
        if error_type in error_type_id:
            return error_type_id[error_type]
        return error_type_id["KEEP"]

    def size(self):
        return self._curr_cpt

    def __len__(self):
        return self.size()


if __name__ == "__main__":

    tagger = TagEncoder()

    from noiser.Noise import Lexicon
    lexicon = Lexicon("../resources/Lexique383.tsv")
