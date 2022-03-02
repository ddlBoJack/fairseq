#!/bin/bash
# set -x

# This block install conda.
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
~/miniconda/bin/conda init $(echo $SHELL | awk -F '/' '{print $NF}')
echo 'Successfully installed miniconda...'
echo -n 'Conda version: '
~/miniconda/bin/conda --version
echo -e '\n'

# This block setup user environment.
source ~/miniconda/etc/profile.d/conda.sh
conda info --env
# echo "which conda: $(which conda)" #using ~/miniconda/bin/conda
# echo "which pip: $(which pip)" #using ~/miniconda/bin/pip
# echo "which python: $(which python)" #using ~/miniconda/bin/python
# echo 'Create fairseq env...'
# conda create -n fairseq -y -q
# conda init $(echo $SHELL | awk -F '/' '{print $NF}')
# source ~/.bashrc
# conda env list
# echo 'Activate fairseq env...'
# conda activate fairseq
# conda env list

# This block install components.
~/miniconda/bin/pip install tensorboardX
~/miniconda/bin/pip install Cython
~/miniconda/bin/pip install soundfile
sudo apt-get -y install libsndfile1
# git clone -b v-ziyangma https://github.com/ddlBoJack/fairseq.git
# cd fairseq
echo "Work directory: $(pwd)"
export PYTHONPATH=/tmp/code:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"
echo "PATH: $PATH"
export PATH=~/.local/bin:$PATH
export PATH=~/miniconda/bin:$PATH
# export PATH=~/miniconda/envs/fairseq/bin:$PATH
echo "PATH: $PATH"
echo 'fairseq install...'
~/miniconda/bin/pip install --editable ./
~/miniconda/bin/python setup.py build_ext --inplace
echo -e '\n'

# source ~/miniconda/etc/profile.d/conda.sh
# echo 'Activate fairseq env...'
# conda activate fairseq

# edit your exp
prefix_dir=/datablob/users/v-ziyangma
model_name=data2vec
exp_name=data2vec_960h_devclean

#edit your config
config_dir=./config/data2vec/audio/pretraining
config_name=base_librispeech

#edit your data
data_path=${prefix_dir}/data/manifest/debug/
train_subset=train_960
valid_subset=dev_clean

# edit your compute resource
distributed_world_size=16
update_freq=[1]
max_tokens=3800000

#edit your ckpt
model_path=${prefix_dir}/model/${model_name}/${exp_name}
mkdir -p ${model_path}

#edit your log
tb_path=${prefix_dir}/log/${model_name}/${exp_name}/tensorboard
mkdir -p ${tb_path}
log_file=${prefix_dir}/log/${model_name}/${exp_name}/hydra_train.log

# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
~/miniconda/bin/python fairseq_cli/hydra_train.py  \
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