#!/bin/bash
export PYTHONPATH=~/github/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/github/fairseq

# edit your exp
model_name=data2vec_jt
exp_name=finetune_debug

#edit your config
config_dir=~/github/fairseq/data2vec_jt/config/finetuning
config_name=base_100h

#edit your data
data_path=~/data/LibriSpeech/manifest/resource/
train_subset=test_clean
valid_subset=test_clean

# edit your compute resource
distributed_world_size=1
update_freq=[2]
max_tokens=1600000

#edit your pretrained model
model_path=/home/v-ziyangma/model/data2vec_jt/pretrain_debug/checkpoint_1_10.pt

#edit your log: !!too slow to write to datablob!!
tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
mkdir -p ${tb_path}
log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log

# set finetune output model
finetuning_output_dir=~/log/${model_name}/${exp_name}/${train_subset}_${valid_subset}
# mkdir -p ${finetuning_output_dir}

# export CUDA_VISIBLE_DEVICES=0,1,2,3
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
common.log_interval=1 \
checkpoint.save_interval_updates=5 \
checkpoint.keep_interval_updates=1 \
dataset.num_workers=1 \
dataset.validate_after_updates=10 \
dataset.validate_interval_updates=10 \
common.wandb_project=debug \
common.user_dir=data2vec_jt \
# common.tensorboard_logdir=${tb_path} \
# common.log_file=${log_file}  \
# checkpoint.reset_dataloader=true \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cd scripts
# python average_checkpoints.py \
#     --inputs /mnt/exp/project/NMT \
#     --num-epoch-checkpoints 10 \
#     --output /mnt/exp/project/NMT