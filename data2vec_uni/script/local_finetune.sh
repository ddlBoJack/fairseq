#!/bin/bash
export PYTHONPATH=~/github/fairseq:$PYTHONPATH
cd ~/github/fairseq

# edit your exp
model_name=data2vec_uni
exp_name=finetune_debug

#edit your config
config_dir=~/github/fairseq/data2vec_uni/config/joint
config_name=debug_finetuning

#edit your data
data_path=~/data/LibriSpeech/manifest/resource/
train_subset=test_clean
valid_subset=test_clean

# edit your compute resource
distributed_world_size=1
update_freq=[2]
max_tokens=1600000

#edit your pretrained model
model_path=/home/v-ziyangma/model/data2vec_uni/pretrain_debug/checkpoint_last.pt

#edit your log: !!too slow to write to datablob!!
# tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log

# set finetune output model
finetuning_output_dir=~/log/${model_name}/${exp_name}/${train_subset}_${valid_subset}
# mkdir -p ${finetuning_output_dir}

echo "Start finetuning!!!"
echo -e '\n'
# python fairseq_cli/hydra_train.py \
python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
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
common.user_dir=data2vec_uni \
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cd scripts
# python average_checkpoints.py \
#     --inputs /mnt/exp/project/NMT \
#     --num-epoch-checkpoints 10 \
#     --output /mnt/exp/project/NMT