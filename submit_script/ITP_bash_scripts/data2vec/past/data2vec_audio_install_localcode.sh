#!/bin/bash
# set -x
cd /datablob/users/v-ziyangma/code/fairseq
export PYTHONPATH=/datablob/users/v-ziyangma/code/fairseq:$PYTHONPATH

sudo chmod 777 -R /miniconda
pip install bitarray
pip install tensorboardX
pip install sacrebleu
pip install Cython
python setup.py build_ext --inplace

# This block setup user environment.
source /miniconda/etc/profile.d/conda.sh
conda env list
echo 'Create fairseq env...'
conda create -n fairseq -y -q
conda init $(echo $SHELL | awk -F '/' '{print $NF}')
source ~/.bashrc
conda env list
echo 'Activate fairseq env...'
conda activate fairseq
conda info --env

# This block install components.
pip install tensorboardX
pip install Cython
pip install soundfile
sudo apt-get -y install libsndfile1
# git clone -b v-ziyangma https://github.com/ddlBoJack/fairseq.git
# cd fairseq
echo "Work directory: $(pwd)"
export PYTHONPATH=${pwd}:$PYTHONPATH
echo "Work directory: $(pwd)"
echo "PATH: $PATH"
export PATH=~/.local/bin:$PATH
echo "PATH: $PATH"
echo 'fairseq install...'
pip install --editable ./
python setup.py build_ext --inplace
echo -e '\n'

# source ~/miniconda/etc/profile.d/conda.sh
# echo 'Activate fairseq env...'
# conda activate fairseq

# edit your exp
prefix_dir=/datablob/users/v-ziyangma
model_name=data2vec
exp_name=data2vec_960h_devclean_test2xG4

#edit your config
config_dir=./config/data2vec/audio/pretraining
config_name=base_librispeech

#edit your data
data_path=${prefix_dir}/data/manifest/debug/
train_subset=train_960
valid_subset=dev_clean

# edit your compute resource
distributed_world_size=8
update_freq=[4]
max_tokens=1900000

#edit your ckpt
model_path=${prefix_dir}/model/${model_name}/${exp_name}
mkdir -p ${model_path}

#edit your log
tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
mkdir -p ${tb_path}
log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log

export PYTHONPATH=${pwd}:$PYTHONPATH
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir ${config_dir}  \
--config-name ${config_name}  \
task.data=${data_path}  \
dataset.train_subset=${train_subset}  \
dataset.valid_subset=${valid_subset}  \
checkpoint.save_dir=${model_path}  \
common.tensorboard_logdir=${tb_path} \
common.log_file=${log_file}  \
distributed_training.distributed_world_size=${distributed_world_size}  \
optimization.update_freq=${update_freq} \
dataset.max_tokens=${max_tokens}

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