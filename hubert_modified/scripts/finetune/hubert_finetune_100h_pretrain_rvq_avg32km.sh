#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=hubert_modified
exp_name=hubert_pretrain_rvq_avg32km_hubertloss
checkpoint=checkpoint_last
finetune=train_100h
model_path=~/models/${model_name}/${exp_name}/${checkpoint}/${finetune}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Start finetuning!!!"
echo -e '\n'
# finetune
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir  /mnt/lustre/sjtu/home/zym22/fairseq/hubert_modified/config/finetune \
--config-name hubert_base_10h  \
checkpoint.save_interval=1 \
checkpoint.keep_last_epochs=-1 \
checkpoint.keep_best_checkpoints=1 \
checkpoint.no_epoch_checkpoints=true \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task.data=/mnt/lustre/sjtu/home/zym22/data/manifest/resource  \
task.label_dir=/mnt/lustre/sjtu/home/zym22/data/manifest/resource \
dataset.train_subset=train_clean_100  \
dataset.valid_subset=dev_other  \
dataset.num_workers=2 \
dataset.max_tokens=3200000  \
dataset.validate_interval=1 \
dataset.validate_after_updates=20000 \
distributed_training.distributed_world_size=4 \
optimization.update_freq=[2] \
optimization.max_update=80000 \
optimization.lr=[0.00003] \
model.w2v_path=/mnt/lustre/sjtu/home/zym22/models/hubert_modified/hubert_pretrain_rvq_avg32km_hubertloss/checkpoint_last.pt \
model.freeze_finetune_updates=10000 \
common.log_interval=200 \
common.log_file=${model_path}/log/hydra_train.log \
common.user_dir=hubert_modified \
common.wandb_project=tribert \
# common.tensorboard_logdir=${model_path}/tensorboard \
# +criterion.wer_kenlm_model=${kenlm_model_path}  \
# +criterion.wer_lexicon=${lexicon_path}  \
# +criterion.wer_lm_weight=2 \
# +criterion.wer_word_score=-1 \
