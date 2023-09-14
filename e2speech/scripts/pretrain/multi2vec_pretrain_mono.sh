#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=e2speech
exp_name=multi2vec_pretrain_ce_mono
model_path=/data/zym22/models/${model_name}/${exp_name}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py \
--config-dir ~/fairseq/multi2vec/config/pretraining  \
--config-name base_librispeech  \
checkpoint.save_interval=20 \
checkpoint.keep_last_epochs=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task._name=multi2vec_pretraining \
task.data=/mnt/lustre/sjtu/home/zym22/data/manifest/resource \
task.normalize=true  \
+task.label_dir=/mnt/lustre/sjtu/home/zym22/data/wav2vec-u \
+task.label_type=phn \
dataset.train_subset=train_960  \
dataset.valid_subset=dev_clean  \
dataset.num_workers=4 \
dataset.max_tokens=1960000 \
dataset.disable_validation=true \
criterion._name=multi2vec \
distributed_training.distributed_world_size=8  \
optimization.update_freq=[4] \
model._name=multi2vec \
common.log_interval=200 \
common.log_file=${model_path}/log/hydra_train.log \
common.wandb_project=multi2vec \
common.user_dir=multi2vec \
+model.ce_loss=true \
+model.mse_loss=true \
# checkpoint.restore_file="checkpoint_last.pt" \
# checkpoint.reset_dataloader=true \
# common.tensorboard_logdir=${model_path}/tensorboard \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path} 