# gramerco

### Download list of french words (with morphologic features)
`wget http://www.lexique.org/databases/Lexique383/Lexique383.tsv`

### Prepare lexicon replacements for (nouns, verbs and adjectives)
`python3 scripts/noise_lex.py resources/Lexique383.tsv FRENCH_TXT -nom -adj -ver > FRENCH_TXT.lex`

It outputs the same texts received as input (FRENCH_TXT) together with noun, adjectives and verbs found on each sentence:

Line | second
-----|-------
Visite du chef d'état-major sud-africain en Arabie saoudite | NOM￨f￨s:visites￨Visite NOM￨-￨s:chefs￨chef NOM￨m￨s:états￨état
Fort séisme aux Célèbes: au moins 13 morts | NOM￨m￨s:séismes￨séisme NOM￨-￨p:mortes￨morte￨mort￨morts
A l'attention des chefs de rubrique sportive | NOM￨f￨s:attentions￨attention NOM￨-￨p:chef￨chefs NOM￨f￨s:rubriques￨rubrique      ADJ￨f￨s:sportifs￨sportives￨sportif￨sportive


### Compute dictionary with word frequencies
`python3 scripts/txt2dic.py < FRENCH_TXT > FRENCH_TXT.dic`

### Compute words that can be appended (pronouns, adverbs, prepositions and punctuation)
`python3 scripts/txt2app.py resources/Lexique383.tsv -pro -adv -pre -pun > FRENCH_TXT.app`

### Generate noisy dataset
```
python3 scripts/add_french_noise.py -dic FRENCH_TXT.dic -rep Lexique383.tsv FRENCH_TXT.lex -app FRENCH_TXT.app > FRENCH_TXT.noise
```
