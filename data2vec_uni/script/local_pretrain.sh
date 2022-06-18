#!/bin/bash
export PYTHONPATH=~/github/fairseq:$PYTHONPATH
cd ~/github/fairseq

# edit your exp
model_name=data2vec_uni
exp_name=pretrain_debug

# edit your config
config_dir=~/github/fairseq/data2vec_uni/config/joint
config_name=debug

# edit your data
data_path=~/data/LibriSpeech/manifest/data2vec_uni/
train_subset=train_clean_100
valid_subset=dev_clean
speech_data=None

# edit your compute resource
distributed_world_size=1
update_freq=[4]
max_tokens=700000

# edit your ckpt
model_path=~/model/${model_name}/${exp_name}
mkdir -p ${model_path}

# edit your pretrained model
text_model_path=/home/v-ziyangma/model/roberta/roberta_phone_pretrain/checkpoint_best.pt
speech_model_path=/home/v-ziyangma/model/data2vec/download_pretrained/audio_base_ls.pt

# edit your log
# tb_path=~/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=~/log/${model_name}/${exp_name}/hydra_train.log


# export CUDA_VISIBLE_DEVICES=1,2

echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python fairseq_cli/hydra_train.py \
python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
task.speech_data=${speech_data}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
checkpoint.save_dir=${model_path}  \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens} \
model.speech_model_path=${speech_model_path} \
model.text_model_path=${text_model_path} \
common.user_dir=data2vec_uni \
checkpoint.reset_dataloader=true
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \


# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cmd
# bash submit_script/local_bash_scripts/data2vec/data2vec_audio.sh