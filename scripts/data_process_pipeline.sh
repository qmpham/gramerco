source ~/anaconda3/bin/activate gramerco-cpu

DATA_DIR=../resources

python noise_lex.py $DATA_DIR/Lexique383.tsv $DATA_DIR/fr_expl.txt > $DATA_DIR/fr_expl.lex
python txt2dic.py < $DATA_DIR/fr_expl.txt > $DATA_DIR/fr_expl.dic
python3 txt2app.py $DATA_DIR/Lexique383.tsv -pro -adv -pre -pun > $DATA_DIR/fr_expl.app
python add_french_noise.py $DATA_DIR/fr_expl.lex -dic $DATA_DIR/fr_expl.dic -rep $DATA_DIR/Lexique383.tsv -app $DATA_DIR/fr_expl.app > $DATA_DIR/fr_expl.noise
python data/generate_dataset.py $DATA_DIR/fr_expl.noise -to $DATA_DIR/data
python data/preprocess_dataset.py $DATA_DIR/data
