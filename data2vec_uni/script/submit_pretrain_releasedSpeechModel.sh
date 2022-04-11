#!/bin/bash
set -x

# edit your exp
prefix_dir=/datablob/users/v-ziyangma
model_name=data2vec_uni
exp_name=data2vec_uni_pretrain_100h_860h_textDoEma_releasedSpeechModel_190w_G8

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
update_freq=[2]
max_tokens=1900000

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
optimization.max_update=100000 \
checkpoint.save_interval_updates=10000 \
checkpoint.keep_interval_updates=10 \
model.loss_beta=0.25 \
common.user_dir=data2vec_uni
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \

cp -r /tmp/code/outputs/ ${prefix_dir}/log/${model_name}/${exp_name}/

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

echo -e '\n'
echo "finshed!"