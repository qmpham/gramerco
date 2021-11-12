import pyonmttok
from transformers import FlaubertTokenizer
import re


class WordTokenizer:
    def __init__(self, cls_tokenizer):
        self.subword_tokenizer = cls_tokenizer.from_pretrained("flaubert/flaubert_base_cased")

    def tokenize(self, text, n_pass=1):
        for i in range(n_pass):
            toks = self._tokenize_single(text)

        return toks

    def unite_tokens(self, tokens):
        text = ' '.join(toks)
        text = re.sub(" '", "'", text)
        return text

    def _tokenize_single(self, text):
        toks_id = self.subword_tokenizer(text)["input_ids"][1:-1]
        toks = [self.subword_tokenizer._convert_id_to_token(tid) for tid in toks_id]
        final_toks = list()
        current_word = ''
        for tok in toks:
            if tok == "'</w>":
                current_word += "'"
            elif tok[-4:] == '</w>':
                final_toks.append(current_word + tok[:-4])
                current_word = ''
            else:
                current_word += tok

        return final_toks


if __name__ == "__main__":

    text = "Je l'appelle Maxime"

    # t = pyonmttok.Tokenizer("conservative", joiner_annotate=False)
    # toks, _ = t.tokenize(text)
    # print(toks)

    # t_flaub = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
    # toks_id = t_flaub(text, padding=True)["input_ids"]
    # toks = [t_flaub._convert_id_to_token(tid) for tid in toks_id]
    # print(toks)
    print(text)
    t_word = WordTokenizer(FlaubertTokenizer)
    toks = t_word.tokenize(text)
    print(toks)
    
