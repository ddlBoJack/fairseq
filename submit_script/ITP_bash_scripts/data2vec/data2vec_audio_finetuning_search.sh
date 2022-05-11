#!/bin/bash
set -x
rm -rf ./outputs/

# edit your exp
prefix_dir=/modelblob/users/v-ziyangma
prefix_dir_wcy=/modelblob/users/v-chengw
model_name=data2vec
exp_name=data2vec_finetuning_search

#edit your config
config_dir=./config/data2vec/audio/finetuning
config_name=base_1h

#edit your data
data_path=${prefix_dir_wcy}/data/librispeech/manifest/resource/
# data_path=/tmp/data/manifest/resource/
train_subset=train_1h
valid_subset=dev_other

# edit your compute resource
distributed_world_size=4
update_freq=[2]
max_tokens=3200000

#edit your pretrained model
checkpoints=(checkpoint_106_100000 checkpoint_212_200000 checkpoint_317_300000 checkpoint_423_400000)

kenlm_model_path=${prefix_dir}/model/language_model/4-gram.bin
lexicon_path=${prefix_dir}/model/language_model/librispeech_lexicon.lst

#edit your log: !!too slow to write to datablob!!
# tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log

for checkpoint in ${checkpoints[*]}; do

    model_path=${prefix_dir}/model/${model_name}/data2vec_checkpoint_per2w/${checkpoint}.pt

    # set finetune output model
    finetuning_output_dir=${prefix_dir}/model/${model_name}/${exp_name}/${checkpoint}/${train_subset}_${valid_subset}
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
    hydra.run.dir=${finetuning_output_dir} \
    distributed_training.distributed_world_size=${distributed_world_size}  \
    optimization.update_freq=${update_freq} \
    dataset.max_tokens=${max_tokens}  \
    task.normalize=true \
    +criterion.wer_kenlm_model=${kenlm_model_path}  \
    +criterion.wer_lexicon=${lexicon_path}  \
    +criterion.wer_lm_weight=2 \
    +criterion.wer_word_score=-1 \
    common.tensorboard_logdir=$AZUREML_TB_PATH
    # common.log_file=${log_file}  \

done

# cp -r /tmp/code/outputs/ ${prefix_dir}/log/${model_name}/${exp_name}/

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cd scripts
# python average_checkpoints.py \
#     --inputs /mnt/exp/project/NMT \
#     --num-epoch-checkpoints 10 \
#     --output /mnt/exp/project/NMT

echo -e '\n'
echo "finshed!"

. ./submit_script/debug/hold_sleep.sh