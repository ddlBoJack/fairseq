#!/bin/bash
export PYTHONPATH=~/github/fairseq:$PYTHONPATH
cd ~/github/fairseq

# edit your exp
model_name=data2vec
exp_name=data2vec_debug
# exp_name=data2vec_debug_finetuning

# edit your config
config_dir=~/github/fairseq/data2vec_uni/config/joint
config_name=debug
# config_name=debug_finetuning

# edit your data
data_path=~/data/LibriSpeech/manifest/data2vec_uni/
train_subset=train_clean_100
valid_subset=train_clean_100

# edit your compute resource
distributed_world_size=1
update_freq=[2]
max_tokens=1000000

# edit your ckpt
model_path=~/model/${model_name}/${exp_name}
mkdir -p ${model_path}

#edit your pretrained model
# model_path=/home/v-ziyangma/model/data2vec/data2vec_debug/checkpoint_1_60.pt

# edit your log
tb_path=~/log/${model_name}/${exp_name}/tensorboard
mkdir -p ${tb_path}
log_file=~/log/${model_name}/${exp_name}/hydra_train.log

# set finetune output model
# finetuning_output_dir=~/log/${model_name}/${exp_name}/${train_subset}_${valid_subset}

export CUDA_VISIBLE_DEVICES=1

echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python fairseq_cli/hydra_train.py -m \
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
dataset.max_tokens=${max_tokens} \
common.user_dir=data2vec_uni

# echo "Start finetuning!!!"
# echo -e '\n'
# finetune
# python fairseq_cli/hydra_train.py  \
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
# --config-dir ${config_dir}  \
# --config-name ${config_name}  \
# task.data=${data_path}  \
# dataset.train_subset=${train_subset}  \
# dataset.valid_subset=${valid_subset}  \
# model.w2v_path=${model_path} \
# hydra.run.dir=${finetuning_output_dir} \
# task.normalize=true \
# distributed_training.distributed_world_size=${distributed_world_size}  \
# optimization.update_freq=${update_freq} \
# dataset.max_tokens=${max_tokens} \
# common.user_dir=data2vec_uni

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cmd
# bash submit_script/local_bash_scripts/data2vec/data2vec_audio.sh