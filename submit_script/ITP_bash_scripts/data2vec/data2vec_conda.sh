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
echo "Conda info: "
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
~/miniconda/bin/pip install editdistance
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
exec bash

# source ~/miniconda/etc/profile.d/conda.sh
# echo 'Activate fairseq env...'
# conda activate fairseq
