#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=emotion2vec
exp_name=audio2_base_libri_cp_iemocap_meld_cmumosei_msppodcast_mead_cls1_clstype_chunk10_warmup5000_lr75e-5
model_path=/mnt/lustre/sjtu/home/zym22/models/${model_name}/${exp_name}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Start pretraining!!!"
echo -e '\n'

# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py \
python fairseq_cli/hydra_train.py \
--config-dir examples/data2vec/config/v2 \
--config-name base_audio_only_task \
checkpoint.save_interval=10 \
checkpoint.keep_last_epochs=-1 \
checkpoint.no_epoch_checkpoints=false \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task.data=/mnt/lustre/sjtu/home/zym22/data/emotion_recognition/manifest \
task.min_sample_size=8000 \
dataset.train_subset=iemocap_meld_cmumosei_msppodcast_mead \
dataset.valid_subset=iemocap_full  \
dataset.num_workers=1 \
dataset.disable_validation=true \
dataset.max_tokens=600000 \
distributed_training.distributed_world_size=4  \
optimization.update_freq=[6] \
optimization.lr=[0.000075] \
optimization.max_update=100000 \
lr_scheduler.warmup_updates=5000 \
model.ema_anneal_end_step=20000 \
model.depth=8 \
model.modalities.audio.prenet_depth=4 \
+model.modalities.audio.num_extra_tokens=10 \
+model.cls_type="chunk" \
+model.cls_loss=1 \
+model.d2v_loss=1 \
common.wandb_project=emotion2vec \
checkpoint.restore_file=/mnt/lustre/sjtu/home/zym22/models/released/data2vec2/base_libri.pt \
checkpoint.reset_optimizer=true \
checkpoint.reset_dataloader=true \
checkpoint.reset_meters=true \
checkpoint.reset_lr_scheduler=true \
hydra.run.dir=${model_path} \
# common.log_file=${model_path}/log/hydra_train.log \
# common.tensorboard_logdir=${model_path}/tensorboard \


# pretrain
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
# python fairseq_cli/hydra_train.py \
# --config-dir ~/fairseq/emotion2vec/config/pretraining  \
# --config-name base_librispeech  \
# checkpoint.save_interval=10 \
# checkpoint.keep_last_epochs=-1 \
# checkpoint.no_epoch_checkpoints=false \
# checkpoint.save_interval_updates=0 \
# checkpoint.keep_interval_updates=-1 \
# checkpoint.save_dir=${model_path}  \
# task._name=emotion2vec_pretraining \
# task.normalize=true  \
# task.max_sample_size=320000 \
# task.min_sample_size=8000 \
# task.data=/mnt/lustre/sjtu/home/zym22/data/emotion_recognition/manifest  \
# dataset.train_subset=iemocap_meld_cmumosei \
# dataset.valid_subset=iemocap_full  \
# dataset.num_workers=4 \
# dataset.max_tokens=1900000 \
# dataset.disable_validation=true \
# criterion._name=emotion2vec \
# distributed_training.distributed_world_size=4  \
# optimization.update_freq=[8] \
# optimization.max_update=10000 \
# optimization.lr=[0.0003] \
# model._name=emotion2vec \
# model.ema_anneal_end_step=1000 \
# common.log_interval=200 \
# common.log_file=${model_path}/log/hydra_train.log \
# +model.ce_loss=false \
# +model.mse_loss=true \
# checkpoint.restore_file=/mnt/lustre/sjtu/home/zym22/models/released/data2vec/audio_base_ls_emotion.pt \
# checkpoint.reset_optimizer=true \
# checkpoint.reset_dataloader=true \
# checkpoint.reset_meters=true \
# checkpoint.reset_lr_scheduler=true \
# common.user_dir=emotion2vec \
# common.wandb_project=emotion2vec \
# common.tensorboard_logdir=${model_path}/tensorboard \
# checkpoint.restore_file="checkpoint_last.pt" \
# checkpoint.reset_dataloader=true \

# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir /home/zym22/model/multi2vec/data2vec_baseline