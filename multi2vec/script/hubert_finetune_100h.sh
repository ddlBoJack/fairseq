#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=multi2vec
exp_name=hubert_baseline
checkpoint=checkpoint700
finetune=train_100h
model_path=/data/zym22/models/${model_name}/${exp_name}/${checkpoint}/${finetune}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

export CUDA_VISIBLE_DEVICES=3,4,6,7
echo "Start finetuning!!!"
echo -e '\n'
# finetune
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir  ~/fairseq/multi2vec/config/finetuning \
--config-name hubert_base_10h  \
checkpoint.save_interval=1 \
checkpoint.keep_last_epochs=-1 \
checkpoint.keep_best_checkpoints=1 \
checkpoint.no_epoch_checkpoints=true \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task.data=/data/zym22/LibriSpeech/manifest/resource  \
task.label_dir=/home/zym22/data/LibriSpeech/manifest/resource \
dataset.train_subset=train_clean_100  \
dataset.valid_subset=dev_other  \
dataset.num_workers=4 \
dataset.max_tokens=3200000  \
dataset.validate_interval=1 \
dataset.validate_after_updates=20000 \
distributed_training.distributed_world_size=4 \
optimization.update_freq=[2] \
optimization.max_update=80000 \
optimization.lr=[0.00003] \
model.w2v_path=/data/zym22/models/multi2vec/hubert_baseline/checkpoint700.pt \
model.freeze_finetune_updates=10000 \
common.log_interval=200 \
common.tensorboard_logdir=${model_path}/tensorboard \
common.log_file=${model_path}/log/hydra_train.log \
common.wandb_project=multi2vec \
# +criterion.wer_kenlm_model=${kenlm_model_path}  \
# +criterion.wer_lexicon=${lexicon_path}  \
# +criterion.wer_lm_weight=2 \
# +criterion.wer_word_score=-1 \
