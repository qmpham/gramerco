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
import logging


def load_data(args):

    train_dataset = GramercoDataset(args.data_path + ".train", args.language)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)

    return train_dataloader


def train(args):
    tagger = TagEncoder()
    train_dataloader = load_data(args)
    criterion = nn.NLLLoss()
    model = GecBertModel(len(tagger))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    losses = list()

    for epoch in tqdm(range(args.n_epochs)):
        train = iter(train_dataloader)
        for batch in train:
            optimizer.zero_grad()

            out = model(**batch["noise_data"])

            logging.info("tag data lens = " + str(batch["tag_data"]["attention_mask"].sum(-1)))
            logging.info("tag out lens = " + str(out["attention_mask"].sum(-1)))

            out = out["tag_out"][out["attention_mask"].bool()]
            tgt = batch["tag_data"]["input_ids"][batch["tag_data"]["attention_mask"].bool()]
            loss = criterion(out, tgt)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

    logging.debug(losses)


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

    parser.add_argument('data_path', help="Input file/s")
    ### optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-log', default="info", help='logging level')
    parser.add_argument('-lang', '--language', default="fr", help='language of the data')
    parser.add_argument('--batch-size', type=int, help='batch size for the training')
    parser.add_argument('--n-epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1, help='Learning rate value.')

    args = parser.parse_args()

    create_logger("stderr", args.log)

    train(args)
