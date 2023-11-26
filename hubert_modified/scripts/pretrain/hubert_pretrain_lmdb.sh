#!/bin/bash
export PYTHONPATH=/mnt/maziyang.mzy/code/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd /mnt/maziyang.mzy/code/fairseq

# pip install lmdb

# edit your exp
model_name=hubert_modified
exp_name=hubert_ils_lmdb_test_parallel
model_path=/mnt/maziyang.mzy/models/${model_name}/${exp_name}
mkdir -p ${model_path}
# mkdir -p ${model_path}/tensorboard
# mkdir -p ${model_path}/log

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

export CUDA_VISIBLE_DEVICES=1
echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python fairseq_cli/hydra_train.py \
python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
--config-dir /mnt/maziyang.mzy/code/fairseq/hubert_modified/config/pretraining \
--config-name hubert_base_librispeech  \
checkpoint.save_interval=10 \
checkpoint.keep_last_epochs=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_interval_updates=20 \
checkpoint.save_dir=${model_path}  \
task._name=hubert_modified_pretraining  \
task.data=/mnt/maziyang.mzy/data/LibriSpeech/manifest/resource \
task.label_dir=/mnt/maziyang.mzy/data/LibriSpeech/wav2vec-u \
task.labels='["phn", "triphn448"]' \
+task.use_lmdb=false \
+task.parallel_lmdb=false \
model._name=hubert_modified \
model.untie_final_proj=false \
+model.ils=true \
+model.ils_layers=[3,11] \
+model.ils_layers_target=[0,1] \
+model.cross_entropy=true \
model.label_rate=50 \
criterion._name=hubert_modified \
dataset.train_subset=train_960_parallel  \
dataset.valid_subset=dev_clean  \
dataset.num_workers=2 \
dataset.max_tokens=700000 \
dataset.validate_interval_updates=20 \
distributed_training.distributed_world_size=1  \
optimization.update_freq=[2] \
common.log_interval=10 \
common.user_dir=hubert_modified \
# common.tensorboard_logdir=${model_path}/tensorboard \
# common.log_file=${model_path}/log/hydra_train.log \
# common.wandb_project=tribert \
# checkpoint.restore_file="checkpoint_last.pt" \
# checkpoint.reset_dataloader=true \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir /home/zym22/model/multi2vec/data2vec_baseline