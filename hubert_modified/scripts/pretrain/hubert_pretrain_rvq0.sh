#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# pip install lmdb

# edit your exp
model_name=hubert_modified
exp_name=hubert_pretrain_rvq0
model_path=~/models/${model_name}/${exp_name}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

# # add temp soft links
# dir_triphn="/mnt/lustre/sjtu/home/zym22/data/wav2vec-u/triphone/hmm-dmm/448"
# dir_phn="/mnt/lustre/sjtu/home/zym22/data/wav2vec-u"
# temp_symlinks=()
# for file in "${dir_triphn}"/*; do
#     file_name=$(basename "${file}")
#     ln -s "${file}" "${dir_phn}/${file_name}"
#     temp_symlinks+=("${dir_phn}/${file_name}")
# done
# trap 'for symlink in "${temp_symlinks[@]}"; do rm -f "${symlink}"; done' EXIT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py \
--config-dir ~/fairseq/hubert_modified/config/pretraining \
--config-name hubert_base_librispeech  \
checkpoint.save_interval=10 \
checkpoint.keep_last_epochs=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task._name=hubert_modified_pretraining  \
task.data=/mnt/lustre/sjtu/home/zym22/data/manifest/resource  \
task.label_dir=/mnt/lustre/sjtu/home/zym22/data/manifest/resource \
task.labels='["rvq0"]' \
model._name=hubert_modified \
model.label_rate=50 \
criterion._name=hubert_modified \
+criterion.hubert_cross_entropy=true \
dataset.train_subset=train_960  \
dataset.valid_subset=dev  \
dataset.num_workers=2 \
dataset.max_tokens=1400000 \
distributed_training.distributed_world_size=8  \
optimization.update_freq=[4] \
common.log_interval=200 \
common.log_file=${model_path}/log/hydra_train.log \
common.user_dir=hubert_modified \
common.wandb_project=tribert \
# common.tensorboard_logdir=${model_path}/tensorboard \
# checkpoint.restore_file="checkpoint_last.pt" \
# checkpoint.reset_dataloader=true \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir /home/zym22/model/multi2vec/data2vec_baseline