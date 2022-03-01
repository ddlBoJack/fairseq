#!/bin/bash
set -x
cd /datablob/users/v-ziyangma/sp_fairseq

model_path=/datablob/users/v-ziyangma/model/data2vec_models/data2vec_debug
data_path=/datablob/users/v-ziyangma/manifest
tensorboard_logdir=/datablob/users/v-ziyangma/tensorboard/data2vec_debug
train_subset=dev_other
valid_subset=dev_clean
distributed_world_size=4
update_freq=16

mkdir -p ${model_path}

python train.py   \
    --distributed-backend 'nccl' \
    --ddp-backend legacy_ddp \
    --distributed-port 29671   \
    --distributed-world-size ${distributed_world_size} \
    --update-freq ${update_freq}   \
    --fp16 \
    --log-format json   \
    --log-interval 200   \
    --tensorboard-logdir ${tensorboard_logdir} \
    --save-dir ${model_path}   \
    --save-interval 5   \
    --save-interval-updates 25000   \
    --keep-interval-updates 1   \
    --no-epoch-checkpoints   \
    --user-dir examples/data2vec \
    --task audio_pretraining   \
    --arch data2vec_audio   \
    ${data_path}   \
    --train-subset ${train_subset} \
    --valid-subset ${valid_subset} \
    --max-sample-size 320000   \
    --min-sample-size 32000   \
    --normalize \
    --num-workers 6  \
    --max-tokens 3800000   \
    --skip-invalid-size-inputs-valid-test   \
    --validate-interval 5   \
    --required-batch-size-multiple 1   \
    --disable-validation   \
    --criterion model   \
    --log-keys '["ema_decay","target_var","pred_var"]'   \
    --max-update 400000   \
    --lr 0.0005   \
    --optimizer adam   \
    --adam-betas '(0.9, 0.98)'   \
    --adam-eps 1e-06   \
    --weight-decay 0.01   \
    --lr-scheduler tri_stage   \
    --phase-ratio '[0.03,0.9,0.07]'   \
    # --model data2vec_audio \
    --extractor-mode layer_norm   \
    --encoder-layerdrop 0.05   \
    --dropout-input 0.0   \
    --dropout-features 0.0   \
    --feature-grad-mult 1.0   \
    --encoder-embed-dim 768   \
    --mask-prob 0.65   \
    --mask-length 10   \
    --loss-beta 0  \
    # --loss-scale null \
    --instance-norm-target-layer \
    --average-top-k-layers 8 \
    # --pos-conv-depth 5 \
    --conv-pos 95   \
    --ema-decay 0.999   \
    --ema-end-decay 0.9999  \
    # --ema-anneal-end-step 30000   \
    --ema-transformer-only \
    --ema-layers-only \
    # --require-same-masks \
    # --mask-dropout 0 \