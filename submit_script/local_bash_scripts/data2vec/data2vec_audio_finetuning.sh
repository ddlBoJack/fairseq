#!/bin/bash
export PYTHONPATH=~/github/fairseq:$PYTHONPATH
cd ~/github/fairseq

# edit your exp
model_name=data2vec
exp_name=data2vec_debug_finetuning

#edit your config
config_dir=~/github/fairseq/config/data2vec/audio/finetuning
config_name=debug

#edit your data
data_path=~/data/LibriSpeech/manifest/resource/
train_subset=test_clean
valid_subset=test_clean

#edit your pretrained model
model_path=/home/v-ziyangma/model/data2vec/data2vec_debug/checkpoint_last.pt

#edit your log
# tb_path=~/log/${model_name}/${exp_name}/tensorboard
# mkdir -p ${tb_path}
# log_file=~/log/${model_name}/${exp_name}/hydra_train.log

# set finetune output model
finetuning_output_dir=~/log/${model_name}/${exp_name}/${train_subset}_${valid_subset}
# mkdir -p ${finetuning_output_dir}

echo "Start finetuning!!!"
echo -e '\n'
# finetuning
# python fairseq_cli/hydra_train.py  \
python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
model.w2v_path=${model_path} \
hydra.run.dir=${finetuning_output_dir} \
task.normalize=true
# distributed_training.distributed_world_size=${distributed_world_size}  \
# optimization.update_freq=${update_freq} \
# dataset.max_tokens=${max_tokens}

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 

# cmd
# bash submit_script/local_bash_scripts/data2vec/data2vec_audio_finetuning.sh