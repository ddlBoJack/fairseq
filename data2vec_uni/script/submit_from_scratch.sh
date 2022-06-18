#!/bin/bash
set -x
rm -rf ./outputs/
pip install wandb
python -m wandb login a7e222c6124a8097a90dc62c0a5d3b8d27d17bfb

# edit your exp
prefix_dir=/modelblob/users/v-ziyangma
model_name=data2vec_uni
exp_name=data2vec_uni_100h_860h_align_phn1_380w_2x32G8_beta0

# edit your config
config_dir=./data2vec_uni/config/joint
config_name=base_librispeech_100h_860h

# edit your data
data_path=${prefix_dir}/data/manifest/data2vec_uni/
train_subset=train_clean_100
valid_subset=dev_clean
speech_data=train_860

# edit your compute resource
distributed_world_size=16
update_freq=[1]
max_tokens=2800000

#edit your ckpt
model_path=${prefix_dir}/model/${model_name}/${exp_name}
mkdir -p ${model_path}

# edit your pretrained model
text_model_path=${prefix_dir}/model/roberta/roberta_phone_pretrain/checkpoint_best.pt
speech_model_path=${prefix_dir}/model/data2vec/download_pretrained/audio_base_ls.pt

#edit your log: !!too slow to write to datablob!!
# tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log
# tb_path=$AZUREML_TB_PATH

echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
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
checkpoint.keep_interval_updates=20 \
model.speech_pretrained_model=false \
model.text_pretrained_model=false \
model.text_teacher=false \
model.text_do_ema=false \
common.wandb_project=data2vec_uni \
common.user_dir=data2vec_uni
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \

# mkdir -p ${prefix_dir}/log/${model_name}/${exp_name}
# cp -r /tmp/code/outputs ${model_path}/
# cp -r $AZUREML_TB_PATH ${model_path}/

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

echo -e '\n'
echo "finshed!"

# . ./submit_script/debug/hold_sleep.sh