#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=e2speech
exp_name=multi2vec_pretrain_ce_mono
checkpoint=checkpoint300
finetune=train_100h
model_path=/data/zym22/models/${model_name}/${exp_name}/${checkpoint}/${finetune}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

export CUDA_VISIBLE_DEVICES=0,1
echo "Start finetuning!!!"
echo -e '\n'
# finetune
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir ~/fairseq/multi2vec/config/finetuning \
--config-name base_100h  \
checkpoint.save_dir=${model_path}  \
task.data=/mnt/lustre/sjtu/home/zym22/data/manifest/resource  \
task.normalize=true \
dataset.train_subset=train_clean_100  \
dataset.valid_subset=dev_other  \
dataset.num_workers=4 \
dataset.max_tokens=3200000  \
dataset.validate_interval=1 \
dataset.validate_after_updates=20000 \
distributed_training.distributed_world_size=2 \
optimization.update_freq=[4] \
optimization.max_update=80000 \
model.w2v_path=/data/zym22/models/e2speech/multi2vec_pretrain_ce_mono/checkpoint300.pt \
model.freeze_finetune_updates=10000 \
common.log_interval=200 \
common.log_file=${model_path}/log/hydra_train.log \
common.wandb_project=multi2vec \
common.user_dir=multi2vec \
# common.tensorboard_logdir=${model_path}/tensorboard \
# checkpoint.save_interval=1 \
# checkpoint.keep_last_epochs=-1 \
# checkpoint.keep_best_checkpoints=-1 \
# checkpoint.no_epoch_checkpoints=true \
# checkpoint.save_interval_updates=0 \
# checkpoint.keep_interval_updates=-1 \
# +criterion.wer_kenlm_model=${kenlm_model_path}  \
# +criterion.wer_lexicon=${lexicon_path}  \
# +criterion.wer_lm_weight=2 \
# +criterion.wer_word_score=-1 \