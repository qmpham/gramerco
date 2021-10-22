# gramerco

The network fine-tunes a BERT-like french language model to perform a gramatical error correction task.

Thus, input is a noisy french text while output is a set of tags (one for each input word) that indicates if the word is correct ($KEEP) or if needs to be corrected following the output tag. The next tags are considered:

Tag | Description
-----------------
$KEEP | No correction is needed
$DELETE | the given token must be deleted
$COPY | the given token must be deleted as a copy of the next word
$SWAP | the given and next tokens must be swapped
$HYPHEN | The given and next tokens must be merged with an hyphen
$MERGE | The given and next tokens must be merged 
$SPLIT | The hyphen appearing in the word must be repaced by a space
$CASE | The case of the first character must be changed

### Download list of french words (with morphologic features)
`wget http://www.lexique.org/databases/Lexique383/Lexique383.tsv`

### Prepare lexicon replacements for (nouns, verbs and adjectives)
`python3 scripts/noise_lex.py resources/Lexique383.tsv FRENCH_TXT -nom -adj -ver > FRENCH_TXT.lex`

It outputs the same texts received as input (FRENCH_TXT) together with noun, adjectives and verbs found on each sentence:

Input string | tokens recognised
-------------|-------
Visite du chef d'état-major sud-africain en Arabie saoudite | NOM￨f￨s:visites￨Visite <br>NOM￨-￨s:chefs￨chef <br>NOM￨m￨s:états￨état
Fort séisme aux Célèbes: au moins 13 morts | NOM￨m￨s:séismes￨séisme <br>NOM￨-￨p:mortes￨morte￨mort￨morts
Pedro Castro Van-Dunem inhumé à Luanda | VER￨m￨s￨par￨pas:inhumés￨inhume￨inhumaient￨inhumée￨inhumer￨inhumons￨inhuma￨inhumait￨inhumé
Le Matif chute dans la perspective d'une hausse des taux européens | VER￨-￨-￨ind￨pre￨3s:chutes￨chute <br>NOM￨f￨s:perspectives￨perspective <br>NOM￨f￨s:hausses￨hausse <br>ADJ￨m￨p:européenne￨européennes￨européen￨européens

### Compute dictionary with word frequencies
`python3 scripts/txt2dic.py < FRENCH_TXT > FRENCH_TXT.dic`

### Compute words that can be appended (pronouns, adverbs, prepositions and punctuation)
`python3 scripts/txt2app.py resources/Lexique383.tsv -pro -adv -pre -pun > FRENCH_TXT.app`

### Generate noisy dataset
```
python3 scripts/add_french_noise.py -dic FRENCH_TXT.dic -rep Lexique383.tsv FRENCH_TXT.lex -app FRENCH_TXT.app > FRENCH_TXT.noise
```
