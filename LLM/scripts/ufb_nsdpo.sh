#!/bin/bash

DIRECTORY_SFT=sagemaker-user/ufb-sft-tiny-mistral_2025-01-09_14-57-43_198794
USE_WANDB=false
WANDB_KEY=SET_VALUE
WANDB_ENTITY=SET_VALUE

MODEL_NAME=tiny-mistral
# MODEL_NAME=llama2-7b-chat-hf
EXP_NAME_ORIG=ufb-sft-$MODEL_NAME

for SEED in 2021 2022 2023 2024 2025
do
    EXP_NAME="${EXP_NAME_ORIG}_${SEED}"
    python3 train.py model=$MODEL_NAME datasets=[tvhh2] \
        seed=$SEED \
        loss=ns_dpo loss.gamma=0.95 loss.current_time=100 loss.beta=0.1 \
        model.archive=.cache/$DIRECTORY_SFT/LATEST/policy.pt \
        exp_name=$EXP_NAME gradient_accumulation_steps=2 eval_every=1000 \
        batch_size=24 eval_batch_size=12 trainer=BasicTrainer sample_during_eval=false \
        ++wandb.key=$WANDB_KEY ++wandb.enabled=$USE_WANDB \
        ++wandb.entity=$WANDB_ENTITY ++wandb.project=nsdpo ++test_dataset=false \
 
done