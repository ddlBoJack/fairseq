#!/bin/bash
export PYTHONPATH=~/github/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/github/fairseq

# edit your exp
model_name=data2vec_jt
exp_name=pretrain_debug

# edit your config
config_dir=~/github/fairseq/data2vec_jt/config/pretraining
config_name=debug

# edit your data
data_path=~/data/LibriSpeech/manifest/data2vec_jt/
train_subset=train_960
valid_subset=train_960
source_data=train_960
target_data=train_960

# edit your compute resource
distributed_world_size=4
update_freq=[16]
max_tokens=600000

# edit your ckpt
model_path=~/model/${model_name}/${exp_name}
mkdir -p ${model_path}

# edit your log
tb_path=~/log/${model_name}/${exp_name}/tensorboard
mkdir -p ${tb_path}
log_file=~/log/${model_name}/${exp_name}/hydra_train.log


export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python fairseq_cli/hydra_train.py \
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
task.source_data=${source_data}  \
task.target_data=${target_data}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
common.tensorboard_logdir=${tb_path} \
common.log_file=${log_file}  \
checkpoint.save_dir=${model_path}  \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens} \
common.user_dir=data2vec_jt \
common.log_interval=1 \
checkpoint.save_interval_updates=1000 \
dataset.num_workers=4 \
checkpoint.reset_dataloader=true \
common.wandb_project=debug \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cmd
# bash submit_script/local_bash_scripts/data2vec/data2vec_audio.sh