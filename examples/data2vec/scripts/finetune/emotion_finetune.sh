#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=emotion2vec
checkpoint_name=audio2_base_libri_cp_iemocap_meld_cmumosei_msppodcast_mead_cls1_clstype_chunk10_warmup5000_lr75e-5
exp_name=finetune_iemocap_val_session5
model_path=/nfs/maziyang.mzy/models/${model_name}/${checkpoint_name}
mkdir -p ${model_path}

export CUDA_VISIBLE_DEVICES=0,1,2,3

# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py \
python fairseq_cli/hydra_train.py \
--config-dir examples/data2vec/config/audio/classification \
--config-name base_classification \
task._name=d2v_audio_classification \
task.data=/nfs/maziyang.mzy/data/iemocap/manifest/finetune_20231224 \
dataset.train_subset=train \
dataset.valid_subset=Session5  \
dataset.validate_interval=1 \
dataset.num_workers=4 \
distributed_training.distributed_world_size=4 \
optimization.update_freq=[2] \
optimization.lr=[2e-3] \
model.model_path=${model_path}/checkpoint_last.pt \
model.prediction_mode=average_before \
model.label_mixup=false \
criterion.log_keys=['correct'] \
checkpoint.save_dir=${model_path}/${exp_name}  \
checkpoint.best_checkpoint_metric=accuracy \
hydra.run.dir=${model_path}/${exp_name} \
common.user_dir=examples/data2vec \
common.wandb_project=emotion2vec \

# checkpoint.save_interval=10 \
# checkpoint.keep_last_epochs=-1 \
# checkpoint.no_epoch_checkpoints=false \
# checkpoint.save_interval_updates=0 \
# checkpoint.keep_interval_updates=-1 \
# checkpoint.save_dir=${model_path}  \
# task.data=/mnt/lustre/sjtu/home/zym22/data/emotion_recognition/manifest \
# task.min_sample_size=8000 \
# dataset.train_subset=iemocap_meld_cmumosei_msppodcast_mead \
# dataset.valid_subset=iemocap_full  \
# dataset.num_workers=1 \
# dataset.disable_validation=true \
# dataset.max_tokens=600000 \
# distributed_training.distributed_world_size=4  \
# optimization.update_freq=[6] \
# optimization.lr=[0.000075] \
# optimization.max_update=100000 \
# lr_scheduler.warmup_updates=5000 \
# model.ema_anneal_end_step=20000 \
# model.depth=8 \
# model.modalities.audio.prenet_depth=4 \
# +model.modalities.audio.num_extra_tokens=10 \
# +model.cls_type="chunk" \
# +model.cls_loss=1 \
# +model.d2v_loss=1 \
# common.wandb_project=emotion2vec \
# checkpoint.restore_file=/mnt/lustre/sjtu/home/zym22/models/released/data2vec2/base_libri.pt \
# checkpoint.reset_optimizer=true \
# checkpoint.reset_dataloader=true \
# checkpoint.reset_meters=true \
# checkpoint.reset_lr_scheduler=true \
# hydra.run.dir=${model_path} \
# common.log_file=${model_path}/log/hydra_train.log \
# common.tensorboard_logdir=${model_path}/tensorboard \