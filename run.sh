source ~/anaconda3/bin/activate gramerco-cpu

python scripts/noise_lex.py resources/Lexique383.tsv resources/fr_expl.txt > resources/fr_expl.lex
python scripts/txt2dic.py < resources/fr_expl.txt > resources/fr_expl.dic
python3 scripts/txt2app.py resources/Lexique383.tsv -pro -adv -pre -pun > resources/fr_expl.app
python scripts/add_french_noise.py resources/fr_expl.lex -dic resources/fr_expl.dic -rep resources/Lexique383.tsv -app resources/fr_expl.app > resources/fr_expl.noise
python scripts/data/generate_dataset.py resources/fr_expl.noise -to resources/data
python scripts/data/preprocess_dataset.py resources/data
