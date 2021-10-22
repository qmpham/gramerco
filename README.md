# gramerco

## Download list of french words (with morphologic features)
 wget http://www.lexique.org/databases/Lexique383/Lexique383.tsv 

## Prepare lexicon replacements for NOM/VER/ADJ 
python3 scripts/noise_lex.py resources/Lexique383.tsv FRENCH_TXT -nom -adj -ver > FRENCH_TXT.lex

## Compute dictionary with word frequencies
python3 scripts/txt2dic.py < FRENCH_TXT > FRENCH_TXT.dic

## Compute words that can be appended
python3 scripts/txt2app.py < FRENCH_TXT > FRENCH_TXT.app

## Generate noisy dataset
python3 scripts/add_french_noise.py -dic FRENCH_TXT.dic -rep resources/Lexique383.tsv FRENCH_TXT.lex -app FRENCH_TXT.app > FRENCH_TXT.noise
