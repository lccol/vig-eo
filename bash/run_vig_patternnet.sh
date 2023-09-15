#!/bin/bash

export COMET_API_KEY=KEY
POSTFIX="vig"
EXP_ID=$$

python run_training.py \
        -n 100 \
        --batch 16 \
        --dataset patternnet \
        --split 0.70,0.15,0.15 \
        --split-seed 47 \
        --comet-logger \
        --lr 0.0001 \
        --seed 1984733 \
        --model-config configs/vig_patternnet.yaml \
        --checkpoint-folder ckpts/vig/patternnet-"${EXP_ID}${POSTFIX}" \
        --exp-name "${EXP_ID}${POSTFIX}"
