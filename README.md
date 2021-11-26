# gramerco

The network fine-tunes a BERT-like french language model to perform a gramatical error correction task.

Thus, input is a noisy french text while output is a set of tags (one for each input word) that indicates wether the word is correct or if needs to be corrected following the output tag. The next tags are considered:

Tag    | Description | Pattern
-------|-------------|--------
·     | Keep current token                             | (X -> X)
$DELETE     | Erase current token                            | (X -> )
$SWAP     | Swap current and next tokens                   | (X Y -> Y X)
$MERGE     | Merge current and next tokens                  | (X Y -> XY)
$HYPHEN     | Merge with an hyphen current and next tokens   | (X Y -> X-Y)
$SPLIT     | Divide current token by the hyphen             | (X-Y -> X Y)
$CASE     | Flip the case of current token first character | (X -> X')
$\<pos>_\<tok> | Replace current token by tok, where pos is the gramatical category of the original token              | (X -> tok)
$APPEND_\<tok> | Append tok to current token                    | (X -> X tok)
$TRANSFORM_\<g-transform> | Apply the g-transform to current token                    | (X -> X')

## Training

```
python scripts/train.py <data_path> --batch-size 64 --n-epochs 20 -lr 0.001 --save ./resources/models
```

## Inference

```
python scripts/infer.py --save ./resources/models < <file_path> > <dest_file>
```
