import pyonmttok
from transformers import FlaubertTokenizer
import os
import sys
pwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(pwd))
sys.path.append(os.path.dirname(os.path.abspath(pwd)))
from noiser.Noise import Spacy
from collections import defaultdict


class TagEncoder:
    def __init__(self,):
        self._id_to_tag = defaultdict(lambda: '·')
        self._tag_to_id = defaultdict(lambda: 0)

    def id_to_tag(self, i):
        return '·'

    def tag_to_id(self, tag):
        return 0

if __name__ == "__main__":

    ...
