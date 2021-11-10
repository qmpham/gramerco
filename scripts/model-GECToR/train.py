from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
import argparse
from ..data.data import load_data


def train(args):
    res = load_data(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    train(args)
