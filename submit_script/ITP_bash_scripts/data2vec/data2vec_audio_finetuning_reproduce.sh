#!/bin/bash
set -x
pip list

# edit your exp
prefix_dir=/datablob/users/v-ziyangma
model_name=data2vec
exp_name=data2vec_960h_repreduce_oldpretrain_newfinetune

#edit your config
config_dir=./config/data2vec/audio/finetuning
config_name=base_100h_reproduce

#edit your data
data_path=${prefix_dir}/data/manifest/finetuning/
train_subset=train_clean_100
valid_subset=dev_other

# edit your compute resource
distributed_world_size=8
update_freq=[1]
max_tokens=3200000

#edit your pretrained model
model_path=${prefix_dir}/model/${model_name}/data2vec_960h_devclean/checkpoint_best.pt

#edit your log: !!too slow to write to datablob!!
# tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log

# set finetune output model
finetuning_output_dir=${prefix_dir}/model/${model_name}/${exp_name}/${train_subset}_${valid_subset}
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
hydra.run.dir=${finetuning_output_dir} \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens}  \
task.normalize=true  \
+criterion.wer_kenlm_model=${kenlm_model_path}  \
+criterion.wer_lexicon=${lexicon_path}  \
+criterion.wer_lm_weight=2 \
+criterion.wer_word_score=-1
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