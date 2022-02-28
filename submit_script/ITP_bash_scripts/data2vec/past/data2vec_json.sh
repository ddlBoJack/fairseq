#!/bin/bash
set -x
export PYTHONPATH=/datablob/users/v-ziyangma/sp_fairseq:$PYTHONPATH
cd /datablob/users/v-ziyangma/sp_fairseq

model_path=/datablob/users/v-ziyangma/model/data2vec_models/data2vec_debug
mkdir -p ${model_path}

python fairseq_cli/hydra_train.py  \
--distributed-backend 'nccl' \
--ddp-backend legacy_ddp \
--distributed-port 29671   \
--arch data2vec_audio   \
--config-dir /datablob/users/v-ziyangma/sp_fairseq/examples/data2vec/config/audio/pretraining \
--config-name base_librispeech \
common.user_dir=examples/data2ve