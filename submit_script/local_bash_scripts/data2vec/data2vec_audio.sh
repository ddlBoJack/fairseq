#!/bin/bash
export PYTHONPATH=~/github/fairseq:$PYTHONPATH
cd ~/github/fairseq

# edit your exp
model_name=data2vec
exp_name=data2vec_debug

#edit your config
config_dir=~/github/fairseq/config/data2vec/audio/pretraining
config_name=debug

#edit your data
data_path=~/data/LibriSpeech/manifest/debug/
train_subset=test-clean
valid_subset=test-clean

# edit your compute resource
distributed_world_size=1
update_freq=[2]
max_tokens=1000000

#edit your ckpt
model_path=~/model/${model_name}/${exp_name}
mkdir -p ${model_path}

#edit your log
tb_path=~/log/${model_name}/${exp_name}/tensorboard
mkdir -p ${tb_path}
log_file=~/log/${model_name}/${exp_name}/hydra_train.log

# pretrain
# python fairseq_cli/hydra_train.py  \
python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
checkpoint.save_dir=${model_path}  \
common.tensorboard_logdir=${tb_path} \
common.log_file=${log_file}  \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens}

# finetune
#TODO: add finetune

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cmd
# bash submit_script/local_bash_scripts/data2vec/data2vec_audio.sh