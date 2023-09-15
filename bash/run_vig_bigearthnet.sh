#!/bin/bash

export COMET_API_KEY=KEY
POSTFIX="vig"
EXP_ID=$$

python run_training.py \
        -n 100 \
        --batch 256 \
        --dataset bigearthnet \
        --comet-logger \
        --lr 0.0001 \
        --seed 19182318 \
        --model-config configs/vig_bigearthnet.yaml \
        --checkpoint-folder ckpts/vig/supervised-"${EXP_ID}${POSTFIX}" \
        --exp-name "${EXP_ID}${POSTFIX}"
