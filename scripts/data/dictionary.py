import os
from collections import Counter
from multiprocessing import Pool

import torch
from fairseq import utils
from fairseq.data import data_utils
from fairseq.file_chunker_utils import Chunker, find_offsets
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line


class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self, encoder, encoder_type,
    ):
        assert encoder_type in ["tag_encoder", "tokenizer"]
        self.encoder = encoder
        self.encoder_type = encoder_type
        self.dic_itt = (
            encoder._id_to_tag if encoder_type == "tag_encoder" else encoder.encoder
        )
        self.dic_tti = (
            encoder._tag_to_id if encoder_type == "tag_encoder" else encoder.decoder
        )
        self.unk_index = (
            0 if self.encoder_type == "tag_encoder" else self.encoder.unk_token
        )
        self.unk_word = self.dic_itt[self.unk_index]
        self.indices = {}

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.dic_itt):
            return self.dic_itt.get(idx, self.unk_index)

    def get_count(self, idx):
        return self.count[idx]

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.dic_itt)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.dic_tti:
            return self.dic_tti[sym]
        return self.unk_index

    def string(self, tensor):
        """Helper for converting a tensor of token indices to a string.
        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(self.string(t) for t in tensor)

        def token_string(i):
            if i == self.unk_index():
                return self.unk_index_string(escape_unk)
            else:
                return self[i]

        sent = separator.join(token_string(i) for i in tensor)

        return sent

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format("unk")
        else:
            return self.unk_index

    #
    # def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
    #     """Sort symbols by frequency in descending order, ignoring special ones.
    #     Args:
    #         - threshold defines the minimum word count
    #         - nwords defines the total number of words in the final dictionary,
    #             including special symbols
    #         - padding_factor can be used to pad the dictionary size to be a
    #             multiple of 8, which is important on some hardware (e.g., Nvidia
    #             Tensor Cores).
    #     """
    #     if nwords <= 0:
    #         nwords = len(self)
    #
    #     new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
    #     new_symbols = self.symbols[: self.nspecial]
    #     new_count = self.count[: self.nspecial]
    #
    #     c = Counter(
    #         dict(
    #             sorted(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
    #         )
    #     )
    #     for symbol, count in c.most_common(nwords - self.nspecial):
    #         if count >= threshold:
    #             new_indices[symbol] = len(new_symbols)
    #             new_symbols.append(symbol)
    #             new_count.append(count)
    #         else:
    #             break
    #
    #     assert len(new_symbols) == len(new_indices)
    #
    #     self.count = list(new_count)
    #     self.symbols = list(new_symbols)
    #     self.indices = new_indices
    #
    #     self.pad_to_multiple_(padding_factor)

    # def bos(self):
    #     """Helper to get index of beginning-of-sentence symbol"""
    #     return self.bos_index
    #
    # def pad(self):
    #     """Helper to get index of pad symbol"""
    #     return self.pad_index
    #
    # def eos(self):
    #     """Helper to get index of end-of-sentence symbol"""
    #     return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    # @classmethod
    # def load(cls, f):
    #     """Loads the dictionary from a text file with the format:
    #     ```
    #     <symbol0> <count0>
    #     <symbol1> <count1>
    #     ...
    #     ```
    #     """
    #     d = cls()
    #     d.add_from_file(f)
    #     return d
    #
    # def add_from_file(self, f):
    #     """
    #     Loads a pre-existing dictionary from a text file and adds its symbols
    #     to this instance.
    #     """
    #     if isinstance(f, str):
    #         try:
    #             with open(PathManager.get_local_path(f), "r", encoding="utf-8") as fd:
    #                 self.add_from_file(fd)
    #         except FileNotFoundError as fnfe:
    #             raise fnfe
    #         except UnicodeError:
    #             raise Exception(
    #                 "Incorrect encoding detected in {}, please "
    #                 "rebuild the dataset".format(f)
    #             )
    #         return
    #
    #     lines = f.readlines()
    #     indices_start_line = self._load_meta(lines)
    #
    #     for line in lines[indices_start_line:]:
    #         try:
    #             line, field = line.rstrip().rsplit(" ", 1)
    #             if field == "#fairseq:overwrite":
    #                 overwrite = True
    #                 line, field = line.rsplit(" ", 1)
    #             else:
    #                 overwrite = False
    #             count = int(field)
    #             word = line
    #             if word in self and not overwrite:
    #                 raise RuntimeError(
    #                     "Duplicate word found when loading Dictionary: '{}'. "
    #                     "Duplicate words can overwrite earlier ones by adding the "
    #                     "#fairseq:overwrite flag at the end of the corresponding row "
    #                     "in the dictionary file. If using the Camembert model, please "
    #                     "download an updated copy of the model file.".format(word)
    #                 )
    #             self.add_symbol(word, n=count, overwrite=overwrite)
    #         except ValueError:
    #             raise ValueError(
    #                 f"Incorrect dictionary format, expected '<token> <cnt> [flags]': \"{line}\""
    #             )
    #
    # def _save(self, f, kv_iterator):
    #     if isinstance(f, str):
    #         PathManager.mkdirs(os.path.dirname(f))
    #         with PathManager.open(f, "w", encoding="utf-8") as fd:
    #             return self.save(fd)
    #     for k, v in kv_iterator:
    #         print("{} {}".format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    # def save(self, f):
    #     """Stores dictionary into a text file"""
    #     ex_keys, ex_vals = self._get_meta()
    #     self._save(
    #         f,
    #         zip(
    #             ex_keys + self.symbols[self.nspecial :],
    #             ex_vals + self.count[self.nspecial :],
    #         ),
    #     )

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(5, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ) -> torch.IntTensor:
        # words = line_tokenizer(line)
        ids = (
            self.encoder([line], return_tensors="pt").input_ids[0]
            if self.encoder_type == "tokenizer"
            else self.encoder.encode_line(line)
        )

        if consumer is not None:
            for i, wid in enumerate(ids):
                consumer(self[i], wid.item())
        return ids

    # @staticmethod
    # def _add_file_to_dictionary_single_worker(
    #     filename, tokenize, eos_word, start_offset, end_offset,
    # ):
    #     counter = Counter()
    #     with Chunker(filename, start_offset, end_offset) as line_iterator:
    #         for line in line_iterator:
    #             for word in tokenize(line):
    #                 counter.update([word])
    #             counter.update([eos_word])
    #     return counter
