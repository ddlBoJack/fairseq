#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=emotion2vec
exp_name=audio_base_ls_cp_iemocap_meld_cmu_cp_adv_en_zh
model_path=/mnt/lustre/sjtu/home/zym22/models/${model_name}/${exp_name}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

export CUDA_VISIBLE_DEVICES=4,5
echo "Start pretraining!!!"
echo -e '\n'
# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py \
--config-dir ~/fairseq/emotion2vec/config/pretraining  \
--config-name base_librispeech  \
checkpoint.save_interval=10 \
checkpoint.keep_last_epochs=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task._name=emotion2vec_pretraining \
task.normalize=true  \
task.max_sample_size=320000 \
task.min_sample_size=8000 \
task.data=/mnt/lustre/sjtu/home/zym22/data/emotion_recognition/manifest  \
+task.adversarial_training=true \
+task.adversarial_subset=zh_m3ed_esd \
dataset.train_subset=en_iemocap_meld_esd \
dataset.valid_subset=m3ed_full  \
dataset.num_workers=4 \
dataset.max_tokens=1900000 \
dataset.disable_validation=true \
criterion._name=emotion2vec \
distributed_training.distributed_world_size=2  \
optimization.update_freq=[16] \
optimization.max_update=10000 \
optimization.lr=[0.0003] \
model._name=emotion2vec \
model.ema_anneal_end_step=1000 \
common.log_interval=200 \
common.log_file=${model_path}/log/hydra_train.log \
+model.ce_loss=false \
+model.mse_loss=true \
+model.adversarial_hidden_dim=128 \
+model.adversarial_weight=0.1 \
checkpoint.restore_file=/mnt/lustre/sjtu/home/zym22/models/emotion2vec/audio_base_ls_cp_iemocap_meld_cmumosei/checkpoint80.pt \
checkpoint.reset_optimizer=true \
checkpoint.reset_dataloader=true \
checkpoint.reset_meters=true \
checkpoint.reset_lr_scheduler=true \
common.user_dir=emotion2vec \
common.wandb_project=emotion2vec \
# common.tensorboard_logdir=${model_path}/tensorboard \
# checkpoint.restore_file="checkpoint_last.pt" \
# checkpoint.reset_dataloader=true \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir /home/zym22/model/multi2vec/data2vec_baseline