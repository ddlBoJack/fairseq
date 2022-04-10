#!/bin/bash
set -x
pip list

# edit your exp
prefix_dir=/datablob/users/v-ziyangma
model_name=data2vec
exp_name=data2vec_960h_repreduce

#edit your config
config_dir=./config/data2vec/audio/pretraining
config_name=reproduce

#edit your data
data_path=${prefix_dir}/data/manifest/debug/
train_subset=train_960
valid_subset=dev_other

# edit your compute resource
distributed_world_size=16
update_freq=[1]
max_tokens=3800000

#edit your ckpt
model_path=${prefix_dir}/model/${model_name}/${exp_name}
mkdir -p ${model_path}

#edit your log: !!too slow to write to datablob!!
# tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log

echo "Start training!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
checkpoint.save_dir=${model_path}  \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens}
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \

cp -r /tmp/code/outputs/ ${prefix_dir}/log/${model_name}/${exp_name}/

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