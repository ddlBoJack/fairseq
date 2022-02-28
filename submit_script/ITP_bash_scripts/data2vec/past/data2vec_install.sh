#!/bin/bash
cd ~
pip install torch torchvision torchaudio
pip install tensorboardX
git clone https://github.com/ddlBoJack/fairseq.git
cd fairseq
git checkout v-ziyangma
export PYTHONPATH=~/fairseq:$PYTHONPATH
pip install --editable ./
pip install Cython
pip install soundfile
python setup.py build_ext --inplace

model_path=/datablob/users/v-ziyangma/model/data2vec_models/data2vec_debug
mkdir -p ${model_path}

# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir ~/fairseq/examples/data2vec/config/audio/pretraining \
--config-name base_librispeech  \
task.data=/datablob/users/v-ziyangma/data/manifest/debug/ \
checkpoint.save_dir=${model_path}