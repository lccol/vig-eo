#!/bin/bash

export COMET_API_KEY=KEY
POSTFIX="vit"
EXP_ID=$$

python run_training.py \
        -n 100 \
        --batch 16 \
        --dataset patternnet \
        --split 0.70,0.15,0.15 \
        --split-seed 47 \
        --comet-logger \
        --lr 0.0001 \
        --seed 3213577347 \
        --model-config configs/vit_tiny_patternnet.yaml \
        --checkpoint-folder ckpts/vit/patternnet-"${EXP_ID}${POSTFIX}" \
        --exp-name "${EXP_ID}${POSTFIX}" \
        --model-class vit
