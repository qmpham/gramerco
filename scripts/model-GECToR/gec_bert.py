from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath("."))
from utils import word_collate


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
        word_ends = x.cpu().apply_(lambda t: self._is_end_of_word(t))
        word_ends = torch.roll(word_ends, 1, -1)
        word_ends[:, 0] = 0
        return torch.cumsum(word_ends, -1)

    def forward(self, **inputs):
        h = self.encoder(**inputs)
        word_index = self._generate_word_index(inputs["input_ids"])
        h_w = word_collate(h.last_hidden_state, word_index)
        attention_mask = word_collate(
            inputs["attention_mask"].unsqueeze(-1), word_index, agregation="max"
        ).squeeze(-1)
        out = self.linear_layer(h_w)
        out = torch.softmax(out, -1)

        return {"tag_out": out, "attention_mask": attention_mask}


if __name__ == "__main__":
    model = GecBertModel(50)

    x = model._tokenize_text(
        [
            "Bonjour, je m'appellerait Maximus. Garderez-vous votre sublimissime mansuétude ?",
            "Erreur du système. Redémarrage imminent...",
        ]
    )

    out = model(**x)
