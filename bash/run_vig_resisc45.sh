#!/bin/bash

export COMET_API_KEY=KEY
POSTFIX="vig"
EXP_ID=$$

python run_training.py \
        -n 100 \
        --batch 16 \
        --dataset resisc45 \
        --comet-logger \
        --lr 0.0001 \
        --seed 47 \
        --model-config configs/vig_resisc45.yaml \
        --checkpoint-folder ckpts/vig/resisc45-"${EXP_ID}${POSTFIX}" \
        --exp-name "${EXP_ID}${POSTFIX}"