#!/bin/bash
set -x
rm -rf ./outputs/
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
sudo `which pip` install wandb
python -m wandb login a7e222c6124a8097a90dc62c0a5d3b8d27d17bfb

# edit your exp
prefix_dir=/modelblob/users/v-ziyangma
prefix_dir_wcy=/modelblob/users/v-chengw
model_name=data2vec_jt
exp_name=data2vec_jt_960h_960h_6text_6share_1add_0kstart_001ctc_finetune_2x16G4-P100-IB

#edit your config
config_dir=./config/data2vec/audio/finetuning
config_name=base_100h

#edit your data
data_path=${prefix_dir_wcy}/data/librispeech/manifest/resource/
# data_path=/tmp/data/manifest/resource/
train_subset=train_clean_100
valid_subset=dev_other

# edit your compute resource
distributed_world_size=8
update_freq=[1]
max_tokens=3200000

#edit your pretrained model
checkpoint=checkpoint_123_100000

kenlm_model_path=${prefix_dir}/model/language_model/4-gram.bin
lexicon_path=${prefix_dir}/model/language_model/librispeech_lexicon.lst

#edit your log: !!too slow to write to datablob!!
# tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log

model_path=${prefix_dir}/model/data2vec_jt/data2vec_jt_960h_960h_6text_6share_1add_0kstart/${checkpoint}.pt

# set finetune output model
finetuning_output_dir=${prefix_dir}/model/${model_name}/${exp_name}/${checkpoint}_${train_subset}_${valid_subset}_viterbi
# mkdir -p ${finetuning_output_dir}

echo "Start finetuning!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
model.w2v_path=${model_path} \
checkpoint.save_dir=${finetuning_output_dir}  \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens}  \
task.normalize=true \
common.wandb_project=data2vec_jt \
common.user_dir=data2vec_jt \
# +criterion.wer_kenlm_model=${kenlm_model_path}  \
# +criterion.wer_lexicon=${lexicon_path}  \
# +criterion.wer_lm_weight=2 \
# +criterion.wer_word_score=-1 \
# hydra.run.dir=${finetuning_output_dir} \
# common.log_file=${log_file}  \
# common.tensorboard_logdir=$AZUREML_TB_PATH \

# cp -r /tmp/code/outputs/ ${prefix_dir}/log/${model_name}/${exp_name}/

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

echo -e '\n'
echo "finshed!"

# . ./submit_script/debug/hold_sleep.sh