#!/bin/bash

DATA=$(cd `dirname $0` && pwd)

wget https://raw.githubusercontent.com/Jess1ca/Conjuncts-span-disambiguation/master/data/ptb/PTB.ext -P $DATA

OFFSET=3915
NUM_TRAIN_SENTENCES=39832
NUM_DEV_SENTENCES=1700
NUM_TEST_SENTENCES=2416

sed -n $OFFSET,$((OFFSET + NUM_TRAIN_SENTENCES - 1))p $DATA/PTB.ext > $DATA/ptb-train.ext
sed -n $((OFFSET + NUM_TRAIN_SENTENCES)),$((OFFSET + NUM_TRAIN_SENTENCES + NUM_DEV_SENTENCES - 1))p $DATA/PTB.ext > $DATA/ptb-dev.ext
sed -n $((OFFSET + NUM_TRAIN_SENTENCES + NUM_DEV_SENTENCES)),$((OFFSET + NUM_TRAIN_SENTENCES + NUM_DEV_SENTENCES + NUM_TEST_SENTENCES - 1))p $DATA/PTB.ext > $DATA/ptb-test.ext

python3 $DATA/clean.py $DATA/ptb-train.ext > $DATA/ptb-train.ext.cleaned
python3 $DATA/clean.py $DATA/ptb-dev.ext > $DATA/ptb-dev.ext.cleaned
python3 $DATA/clean.py $DATA/ptb-test.ext > $DATA/ptb-test.ext.cleaned
