#!/bin/bash
export PYTHONPATH=~/github/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/github/fairseq

# edit your exp
model_name=multi2vec
exp_name=pretrain_debug
model_path=~/model/${model_name}/${exp_name}
mkdir -p ${model_path}

# export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py \
--config-dir /home/v-ziyangma/github/fairseq/multi2vec/config/pretraining  \
--config-name multi2vec_base_librispeech  \
common.log_interval=1 \
common.wandb_project=debug \
common.user_dir=multi2vec \
checkpoint.save_interval_updates=10 \
checkpoint.save_dir=${model_path}  \
distributed_training.distributed_world_size=1  \
task._name=multi2vec_pretraining \
task.data=/home/v-ziyangma/data/LibriSpeech/manifest/resource  \
task.label_dir=/home/v-ziyangma/data/LibriSpeech/manifest/resource  \
task.labels=[km] \
task.label_rate=50 \
task.normalize=true  \
dataset.train_subset=dev_clean  \
dataset.valid_subset=dev_clean  \
dataset.num_workers=1 \
dataset.max_tokens=700000 \
criterion._name=multi2vec  \
optimization.update_freq=[2] \
model._name=multi2vec \
# checkpoint.restore_file="checkpoint_last.pt" \
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \
# checkpoint.reset_dataloader=true \