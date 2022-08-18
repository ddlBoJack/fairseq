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
valid_subset=val_10

# edit your compute resource
distributed_world_size=1
update_freq=[4]
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
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
checkpoint.save_dir=${model_path}  \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens} \
common.user_dir=data2vec_jt \
common.log_interval=1 \
checkpoint.save_interval_updates=10 \
dataset.num_workers=1 \
common.wandb_project=debug \
# checkpoint.restore_file="checkpoint_last.pt" \
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \
# checkpoint.reset_dataloader=true \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cmd
# bash submit_script/local_bash_scripts/data2vec/data2vec_audio.sh