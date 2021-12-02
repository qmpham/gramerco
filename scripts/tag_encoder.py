from collections import defaultdict
import torch
from noiser.Noise import Spacy
from transformers import FlaubertTokenizer
import os
import sys
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

    def size(self):
        return self._curr_cpt

    def __len__(self):
        return self.size()


if __name__ == "__main__":

    tagger = TagEncoder()

    from noiser.Noise import Lexicon
    lexicon = Lexicon("../resources/Lexique383.tsv")
