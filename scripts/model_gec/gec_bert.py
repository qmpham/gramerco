from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import sys
import os
pwd = os.path.dirname(__file__)
print("pwd", pwd)
sys.path.append(os.path.dirname(pwd))
sys.path.append(os.path.dirname(os.path.abspath(pwd)))
from data.data import GramercoDataset
try:
    from utils import word_collate
except:
    from .utils import word_collate


class GecBertModel(nn.Module):
    def __init__(self, num_tag, encoder_name="flaubert/flaubert_base_cased"):
        super(GecBertModel, self).__init__()

        self.tokenizer = FlaubertTokenizer.from_pretrained(encoder_name)
        self.encoder = FlaubertModel.from_pretrained(encoder_name)
        self.num_tag = num_tag
        h_size = self.encoder.attentions[0].out_lin.out_features
        self.linear_layer = nn.Linear(h_size, num_tag)

    def _tokenize_text(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True)

    def _is_end_of_word(self, idx):
        tok = self.tokenizer._convert_id_to_token(idx)
        return tok[-4:] in ["</w>", "<s>", "</s>"]

    def _generate_word_index(self, x):
        word_ends = x.clone().cpu().apply_(lambda t: self._is_end_of_word(t))
        word_ends = torch.roll(word_ends, 1, -1)
        word_ends[:, 0] = 0
        return torch.cumsum(word_ends, -1)

    def forward(self, **inputs):
        h = self.encoder(**inputs)

        word_index = self._generate_word_index(inputs["input_ids"])

        h_w = word_collate(h.last_hidden_state, word_index)

        attention_mask_larger = word_collate(
            inputs["attention_mask"].unsqueeze(-1), word_index, agregation="max"
        ).squeeze(-1)
        attention_mask = torch.zeros_like(attention_mask_larger)
        attention_mask[:, 1:-1] = attention_mask_larger[:, 2:]
        out = self.linear_layer(h_w)
        out = torch.softmax(out, -1)
        return {"tag_out": out, "attention_mask": attention_mask}


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
    create_logger('stderr', 'DEBUG')
    model = GecBertModel(50).train()

    path_to_bin = "/home/bouthors/workspace/gramerco-repo/gramerco/resources/bin/data.train"

    bsz = 4
    dataset = GramercoDataset(path_to_bin, 'fr')
    dataloader = DataLoader(dataset, shuffle=True, batch_size=bsz)

    batch = next(iter(dataloader))
    out = model(**batch["noise_data"])
    for i in range(bsz):
        x = batch["noise_data"]["input_ids"][i]
        m = batch["noise_data"]["attention_mask"][i]
        logging.debug(' '.join([model.tokenizer._convert_id_to_token(s.item()) for s in x[m==1]]))
    logging.info("noise data lens = " + str(batch["noise_data"]["attention_mask"].sum(-1)))
    logging.info("tag data lens = " + str(batch["tag_data"]["attention_mask"].sum(-1)))
    logging.info("tag out lens = " + str(out["attention_mask"].sum(-1)))
    logging.info("over")
