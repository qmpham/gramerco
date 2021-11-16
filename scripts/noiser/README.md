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
Le￨· président￨· français￨\$APPEND_se rendais￨\$TRANSF\_VER￨-￨-￨ind￨pre￨3s à￨· moscou￨\$CASE |
