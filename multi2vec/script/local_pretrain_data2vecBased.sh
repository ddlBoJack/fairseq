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
# python fairseq_cli/hydra_train.py \
python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
--config-dir /home/v-ziyangma/github/fairseq/multi2vec/config/pretraining  \
--config-name base_librispeech  \
common.log_interval=1 \
common.wandb_project=debug \
common.user_dir=multi2vec \
checkpoint.save_interval_updates=10 \
checkpoint.save_dir=${model_path}  \
distributed_training.distributed_world_size=1  \
optimization.update_freq=[2] \
task._name=multi2vec_pretraining \
task.data=/home/v-ziyangma/data/LibriSpeech/manifest/resource  \
task.normalize=true  \
+task.labels=km \
dataset.train_subset=dev_clean  \
dataset.valid_subset=dev_clean  \
dataset.num_workers=1 \
dataset.max_tokens=700000 \
model._name=multi2vec \
# checkpoint.restore_file="checkpoint_last.pt" \
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \
# checkpoint.reset_dataloader=true \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 