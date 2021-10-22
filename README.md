# gramerco

* Download list of french words with morph
wget http://www.lexique.org/databases/Lexique383/Lexique383.tsv

* 
python3 scripts/noise_lex.py resources/Lexique383.tsv FRENCH_TXT -nom -adj -ver > FRENCH_TXT.lex
python3 scripts/txt2dic.py < FRENCH_TXT > FRENCH_TXT.dic
python3 scripts/txt2app.py < FRENCH_TXT > FRENCH_TXT.app
python3 scripts/add_french_noise.py -dic FRENCH_TXT.dic -rep resources/Lexique383.tsv FRENCH_TXT.lex -app FRENCH_TXT.app > FRENCH_TXT.noise
