#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=tribert
exp_name=u_triphone448_hmmdmm
model_path=~/models/${model_name}/${exp_name}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py \
--config-dir ~/fairseq/tribert/config/pretraining  \
--config-name hubert_base_librispeech  \
checkpoint.save_interval=10 \
checkpoint.keep_last_epochs=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task.data=/mnt/lustre/sjtu/home/zym22/data/manifest/resource  \
task.label_dir=/mnt/lustre/sjtu/home/zym22/data/wav2vec-u/triphone/hmm-dmm \
task.labels='["triphn"]' \
model.label_rate=50 \
dataset.train_subset=train_960  \
dataset.valid_subset=dev_clean  \
dataset.num_workers=2 \
dataset.max_tokens=1400000 \
distributed_training.distributed_world_size=8  \
optimization.update_freq=[4] \
common.log_interval=100 \
common.tensorboard_logdir=${model_path}/tensorboard \
common.log_file=${model_path}/log/hydra_train.log \
common.wandb_project=tribert \
# checkpoint.restore_file="checkpoint_last.pt" \
# checkpoint.reset_dataloader=true \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir /home/zym22/model/multi2vec/data2vec_baseline