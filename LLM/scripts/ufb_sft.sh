#!bin/bash

USE_WANDB=false
WANDB_KEY=SET_VALUE
WANDB_ENTITY=SET_VALUE

MODEL_NAME=tiny-mistral
# MODEL_NAME=llama2-7b-chat-hf
EXP_NAME=ufb-2rm-sft-$MODEL_NAME


python3 train.py model=$MODEL_NAME datasets=[ufb-2rm] \
    loss=sft exp_name=$EXP_NAME gradient_accumulation_steps=2 \
    batch_size=24 eval_batch_size=24 trainer=BasicTrainer sample_during_eval=false \
    ++wandb.key=$WANDB_KEY ++wandb.enabled=$USE_WANDB \
    ++wandb.entity=$WANDB_ENTITY ++wandb.project=nsdpo ++test_dataset=false \

