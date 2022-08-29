#!/bin/bash
set -x
rm -rf ./outputs/
pip install wandb
python -m wandb login a7e222c6124a8097a90dc62c0a5d3b8d27d17bfb

# edit your exp
prefix_dir=/modelblob/users/v-ziyangma
prefix_dir_wcy=/modelblob/users/v-chengw
model_name=data2vec_uni
exp_name=data2vec_uni_100h_860h_align_phn1_280w_2x32G8_beta0_finetune

#edit your config
config_dir=./data2vec_uni/config/joint
config_name=base_100h

#edit your data
data_path=${prefix_dir}/data/manifest/resource/
# data_path=${prefix_dir}/data/manifest/finetuning/
train_subset=train_clean_100
valid_subset=dev_other

# edit your compute resource
distributed_world_size=8
update_freq=[1]
max_tokens=3200000

#edit your pretrained model
checkpoint=checkpoint_156_200000
model_path=${prefix_dir}/model/${model_name}/data2vec_uni_100h_860h_align_phn1_380w_2x32G8_beta0/${checkpoint}.pt
#edit your log: !!too slow to write to datablob!!
# tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log
# tb_path=$AZUREML_TB_PATH

# set finetune output model
finetuning_output_dir=${prefix_dir}/model/${model_name}/${exp_name}/${checkpoint}_${train_subset}_${valid_subset}_viterbi_alignPhn1_beta0
# mkdir -p ${finetuning_output_dir}

kenlm_model_path=${prefix_dir}/model/language_model/4-gram.bin
lexicon_path=${prefix_dir}/model/language_model/librispeech_lexicon.lst

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
common.wandb_project=data2vec_uni \
common.user_dir=data2vec_uni \
# +criterion.wer_kenlm_model=${kenlm_model_path}  \
# +criterion.wer_lexicon=${lexicon_path}  \
# +criterion.wer_lm_weight=2 \
# +criterion.wer_word_score=-1 \
# hydra.run.dir=${finetuning_output_dir} \
# common.tensorboard_logdir=${tb_path}
# common.log_file=${log_file}  \

# cp -r /tmp/code/outputs ${prefix_dir}/model/${model_name}/${exp_name}/
# cp -r $AZUREML_TB_PATH ${finetuning_output_dir}/

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 


echo -e '\n'
echo "finshed!"

# . ./submit_script/debug/hold_sleep.sh