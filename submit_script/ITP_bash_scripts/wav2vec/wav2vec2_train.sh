#!/bin/bash
set -x

# edit your exp
prefix_dir=/datablob/users/v-ziyangma
model_name=wav2vec2
exp_name=wav2vec2_960h_devclean

#edit your config
config_dir=./examples/wav2vec/config/pretraining
config_name=wav2vec2_base_librispeech

#edit your data
data_path=${prefix_dir}/data/manifest/debug/
train_subset=train_960
valid_subset=dev_clean

# edit your compute resource
distributed_world_size=8
update_freq=[8]

#edit your ckpt
model_path=${prefix_dir}/model/${model_name}/${exp_name}
mkdir -p ${model_path}

#edit your log
tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
mkdir -p ${tb_path}
log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log

echo "Start training!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
~/miniconda/bin/python fairseq_cli/hydra_train.py  \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
checkpoint.save_dir=${model_path}  \
common.tensorboard_logdir=${tb_path} \
common.log_file=${log_file}  \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq}

# finetune
#TODO: add finetune

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cd scripts
# python average_checkpoints.py \
#     --inputs /mnt/exp/project/NMT \
#     --num-epoch-checkpoints 10 \
#     --output /mnt/exp/project/NMT

echo -e '\n'
echo "finshed!"