#!/bin/bash
set -x

# edit your exp
prefix_dir=/datablob/users/v-ziyangma
model_name=roberta
exp_name=roberta_phone_pretrain

#edit your config
config_dir=./config/roberta
config_name=phone_base

#edit your data
data_path=${prefix_dir}/data/roberta/data-bin

# edit your compute resource
distributed_world_size=8

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
checkpoint.save_dir=${model_path}  \
common.tensorboard_logdir=${tb_path} \
common.log_file=${log_file}  \
distributed_training.distributed_world_size=${distributed_world_size}
# reference: 
# https://github.com/pytorch/fairseq/tree/main/examples/roberta
# https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.pretraining.md
# https://github.com/pytorch/fairseq/tree/main/examples/language_model

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