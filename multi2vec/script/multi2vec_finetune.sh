#!/bin/bash
export PYTHONPATH=/home/zym22/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd /home/zym22/fairseq

# edit your exp
model_name=multi2vec
exp_name=multi2vec_baseline
checkpoint=checkpoint60
finetune=train_10h
model_path=/home/zym22/model/${model_name}/${exp_name}/${checkpoint}/${finetune}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

export CUDA_VISIBLE_DEVICES=6,7
echo "Start finetuning!!!"
echo -e '\n'
# finetune
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir /home/zym22/fairseq/multi2vec/config/finetuning \
--config-name base_10h  \
checkpoint.save_interval=40 \
checkpoint.keep_last_epochs=-1 \
checkpoint.no_epoch_checkpoints=true \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task.data=/home/zym22/data/LibriSpeech/manifest/resource  \
task.normalize=true \
dataset.train_subset=train_10h  \
dataset.valid_subset=dev_other  \
dataset.num_workers=4 \
dataset.max_tokens=3200000  \
dataset.validate_interval=5 \
dataset.validate_after_updates=0 \
distributed_training.distributed_world_size=2  \
optimization.update_freq=[4] \
optimization.max_update=20000 \
model.w2v_path=/home/zym22/model/multi2vec/multi2vec_baseline/checkpoint60.pt \
model.freeze_finetune_updates=0 \
common.log_interval=100 \
common.tensorboard_logdir=${model_path}/tensorboard \
common.log_file=${model_path}/log/hydra_train.log \
common.wandb_project=multi2vec \
common.user_dir=multi2vec \
# +criterion.wer_kenlm_model=${kenlm_model_path}  \
# +criterion.wer_lexicon=${lexicon_path}  \
# +criterion.wer_lm_weight=2 \
# +criterion.wer_word_score=-1 \