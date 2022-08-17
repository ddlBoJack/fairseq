#!/bin/bash
set -x
rm -rf ./outputs/
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
sudo `which pip` install wandb
python -m wandb login a7e222c6124a8097a90dc62c0a5d3b8d27d17bfb

# edit your exp
prefix_dir=/modelblob/users/v-ziyangma
model_name=data2vec_jt
exp_name=data2vec_jt_960h_960h_6text_6share_1add_0kstart_01ctc

# edit your config
config_dir=./data2vec_jt/config/pretraining
config_name=base_librispeech

# edit your data
data_path=${prefix_dir}/data/manifest/data2vec_jt/
train_subset=train_960
valid_subset=val_10

# edit your compute resource
# distributed_world_size=16
update_freq=[4]
max_tokens=700000

# edit your ckpt
model_path=${prefix_dir}/model/${model_name}/${exp_name}
mkdir -p ${model_path}

# edit your log
# tb_path=~/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=~/log/${model_name}/${exp_name}/hydra_train.log

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
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens} \
common.log_interval=200 \
checkpoint.keep_interval_updates=40 \
checkpoint.save_interval_updates=10000 \
common.wandb_project=data2vec_jt \
common.user_dir=data2vec_jt \
# distributed_training.distributed_world_size=${distributed_world_size}  \
# checkpoint.reset_dataloader=true \
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

echo -e '\n'
echo "finshed!"

# cmd
# bash submit_script/local_bash_scripts/data2vec/data2vec_audio.sh