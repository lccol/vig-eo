#!/bin/bash

export COMET_API_KEY=KEY
POSTFIX="vit-tiny"
EXP_ID=$$

python run_training.py \
        -n 100 \
        --batch 256 \
        --dataset bigearthnet \
        --comet-logger \
        --lr 0.0001 \
        --seed 3213577347 \
        --model-config configs/vit_tiny_bigearthnet.yaml \
        --checkpoint-folder ckpts/vit/bigearthnet-"${EXP_ID}${POSTFIX}" \
        --exp-name "${EXP_ID}${POSTFIX}" \
        --model-class vit
