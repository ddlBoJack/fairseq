#!/bin/bash
# export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=INFO

# cd ~/fairseq
python -m wandb login a7e222c6124a8097a90dc62c0a5d3b8d27d17bfb

# edit your exp
model_name=hubert_modified
exp_name=ils_11_monophn40_11_triphn448
model_path=/data/volume1/maziyang.mzy/models/${model_name}/${exp_name}
mkdir -p ${model_path}
# mkdir -p ${model_path}/tensorboard
# mkdir -p ${model_path}/log

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py \
--config-dir hubert_modified/config/pretraining  \
--config-name hubert_base_librispeech  \
checkpoint.save_interval=10 \
checkpoint.keep_last_epochs=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task._name=hubert_modified_pretraining  \
task.data=/data/volume1/maziyang.mzy/LibriSpeech/pai_lmdb  \
task.label_dir=/data/volume1/maziyang.mzy/LibriSpeech/wav2vec-u \
task.labels='["phn", "triphn448"]' \
+task.use_lmdb=true \
+task.parallel_lmdb=true \
model._name=hubert_modified \
model.untie_final_proj=false \
+model.ils=true \
+model.ils_layers=[11,11] \
+model.ils_layers_target=[0,1] \
model.label_rate=50 \
criterion._name=hubert_modified \
dataset.train_subset=train_960_parallel  \
dataset.valid_subset=dev_clean  \
dataset.num_workers=0 \
dataset.max_tokens=2800000 \
distributed_training.distributed_world_size=8  \
optimization.update_freq=[2] \
common.log_format=simple \
common.log_interval=100 \
common.user_dir=hubert_modified \
common.wandb_project=tribert \
# common.log_file=${model_path}/log/hydra_train.log \
# common.tensorboard_logdir=${model_path}/tensorboard \
# checkpoint.restore_file="checkpoint_last.pt" \
# checkpoint.reset_dataloader=true \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir /home/zym22/model/multi2vec/data2vec_baseline