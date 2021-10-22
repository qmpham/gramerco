# gramerco

The network fine-tunes a BERT-like french language model to perform a gramatical error correction task.

Thus, input is a noisy french text while output is a set of tags (one for each input word) that indicates wether the word is correct or if needs to be corrected following the output tag. The next tags are considered:

Tag          | Description
-------------|------------
$KEEP        | No correction is needed (X => X)
$DELETE      | Current token must be deleted (X => )
$SWAP        | Current and next tokens must be swapped (X Y => Y X)
$MERGE       | Current and next tokens must be merged joined (X Y => XY)
$HYPHEN      | Current and next tokens must be merged with an hyphen (X Y => X-Y)
$SPLIT       | Current token has an hyphen to be replaced by a space (X-Y => X Y)
$CASE        | Current token first character must flip its case (X => X')
$TRANSF_tag  | Current token must be inflected according to tag (X => X')
$REPLACE_tok | Replace current token by tok (X => tok)
$APPEND_tok  | Append tok to current token (X => X tok)

### Download list of french words (with morphologic features)
`wget http://www.lexique.org/databases/Lexique383/Lexique383.tsv`

### Prepare lexicon replacements for (nouns, verbs and adjectives)
`python3 scripts/noise_lex.py resources/Lexique383.tsv FRENCH_TXT > FRENCH_TXT.lex`

It outputs the same texts received as input (FRENCH_TXT) together with information of nouns, adjectives and verbs found on each sentence. For instance, given the french sentence 'Le président français se rend à Moscou':

Token | Adj/Noun/Verb found
------|-------
Le    |
président | NOM￨m￨s:présidentes￨présidente￨présidents￨président
français | ADJ￨m￨-:française￨françaises￨français
se | 
rend | VER￨-￨-￨ind￨pre￨3s:rendus￨rendiez￨...￨rendaient￨rend
à | 
Moscou | 

### Compute dictionary with word frequencies
`python3 scripts/txt2dic.py < FRENCH_TXT > FRENCH_TXT.dic`

### Compute words that can be appended (pronouns, adverbs, prepositions and punctuation)
`python3 scripts/txt2app.py resources/Lexique383.tsv -pro -adv -pre -pun > FRENCH_TXT.app`

### Generate noisy dataset
```
python3 scripts/add_french_noise.py -dic FRENCH_TXT.dic -rep Lexique383.tsv FRENCH_TXT.lex -app FRENCH_TXT.app > FRENCH_TXT.noise
```

Noise examples |
-------------- |
Le￨$KEEP président￨$KEEP français￨$APPEND_se rendais￨$TRANSF_VER￨-￨-￨ind￨pre￨3s à￨$KEEP moscou￨$CASE |

