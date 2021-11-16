source ~/anaconda3/bin/activate gramerco

DATA_NAME=AFP
DATA_DIR=../resources
DATA_RAW=$DATA_DIR/$DATA_NAME/$DATA_NAME-raw
DATA_LEX=$DATA_DIR/$DATA_NAME/$DATA_NAME-lex
DATA_NOISE=$DATA_DIR/$DATA_NAME/$DATA_NAME-noise
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin

mkdir -p $DATA_LEX
mkdir -p $DATA_NOISE
mkdir -p $DATA_BIN

python noiser/noise_lex.py $DATA_DIR/Lexique383.tsv $DATA_RAW/*.txt > $DATA_LEX/$DATA_NAME.lex
python noiser/txt2dic.py < $DATA_RAW/*.txt > $DATA_LEX/$DATA_NAME.dic
python noiser/txt2app.py $DATA_DIR/Lexique383.tsv -pro -adv -pre -pun > $DATA_LEX/lexique.app
python noiser/add_french_noise.py $DATA_LEX/$DATA_NAME.lex -dic $DATA_LEX/$DATA_NAME.dic -rep $DATA_DIR/Lexique383.tsv -app $DATA_LEX/lexique.app > $DATA_LEX/$DATA_NAME.noise
python data/generate_dataset.py $DATA_LEX/$DATA_NAME.noise -to $DATA_NOISE/$DATA_NAME
python data/preprocess_dataset.py $DATA_NOISE/$DATA_NAME -log debug -split train -to $DATA_BIN/$DATA_NAME
