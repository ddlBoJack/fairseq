#!/bin/bash
set -x
source ~/miniconda/etc/profile.d/conda.sh
conda env list
echo 'Create fairseq env...'
conda create -n fairseq python=3.7
conda init $(echo $SHELL | awk -F '/' '{print $NF}')
conda env list
echo 'Activate fairseq env...'
conda activate fairseq
conda info --env
echo 'fairseq install...'
pip install --editable ./
echo -e '\n'
exec bash