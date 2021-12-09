#!/bin/bash
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

#cat data/*.txt | python3 scripts/txt2dic.py -minf 5 > french.dic 2> log.dic &
#ls data/*.txt | parallel -j 32 "python3 scripts/noise_lex.py -adj -nom -ver Lexique383.tsv {} > {}.lex 2> {}.log" &> log.parallel &
#ls data/HEAD*.lex | parallel -j 32 "python3 scripts/add_french_noise.py -dic french.dic -rep Lexique383.tsv -app Lexique383.app {} > {}.noise" &> log.parallel &

# python noiser/txt2app.py $DATA_DIR/Lexique383.tsv -pro -adv -pre -pun > $DATA_LEX/lexique.app
# echo "app done"
#
# cat $DATA_RAW/*.txt | python noiser/txt2dic.py -minf 5 > $DATA_LEX/$DATA_NAME.dic 2> $DATA_LEX/log.dic &
# echo "dic done"
#
# ls $DATA_RAW/*.txt | parallel -j 32 'python noiser/noise_lex.py ../resources/Lexique383.tsv {} > ../resources/AFP/AFP-lex/{/.}.lex' 2> $DATA_LEX/log.parallel &
# echo "noise lex done"

# ls $DATA_LEX/*.lex | parallel -j 32 'python noiser/add_french_noise.py -dic ../resources/AFP/AFP-lex/AFP.dic -rep ../resources/Lexique383.tsv -app ../resources/AFP/AFP-lex/lexique.app {} --out ../resources/AFP/AFP-lex/{/.}.noise' &> $DATA_LEX/log.parallel2
#
# echo "add noise done"

# cat $DATA_LEX/*.noise > $DATA_NAME.noise

# python data/generate_dataset.py $DATA_LEX/$DATA_NAME.noise -to $DATA_NOISE/$DATA_NAME

# python data/split.py $DATA_NOISE/$DATA_NAME -dev 0.002 -test 0.01

# python data/preprocess.py $DATA_NOISE/$DATA_NAME.train -log info -lex $DATA_DIR/Lexique383.tsv -app $DATA_LEX/lexique.app --num-workers 16 -out $DATA_BIN/$DATA_NAME.train
python data/preprocess.py $DATA_NOISE/$DATA_NAME.dev -log info -lex $DATA_DIR/Lexique383.tsv -app $DATA_LEX/lexique.app --num-workers 16 -out $DATA_BIN/$DATA_NAME.dev
# python data/preprocess.py $DATA_NOISE/$DATA_NAME.test -log info -lex $DATA_DIR/Lexique383.tsv -app $DATA_LEX/lexique.app --num-workers 16 -out $DATA_BIN/$DATA_NAME.test
