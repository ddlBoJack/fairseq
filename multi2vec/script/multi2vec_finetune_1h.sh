#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=multi2vec
exp_name=multi2vec_baseline
checkpoint=checkpoint700
finetune=train_1h_dev_clean
model_path=/data/zym22/models/${model_name}/${exp_name}/${checkpoint}/${finetune}
mkdir -p ${model_path}
mkdir -p ${model_path}/tensorboard
mkdir -p ${model_path}/log

export CUDA_VISIBLE_DEVICES=6,7
echo "Start finetuning!!!"
echo -e '\n'
# finetune
# python -m debugpy --listen 5678 --wait-for-client fairseq_cli/hydra_train.py  \
python fairseq_cli/hydra_train.py  \
--config-dir ~/fairseq/examples/wav2vec/config/finetuning \
--config-name base_1h  \
checkpoint.save_interval=1 \
checkpoint.keep_last_epochs=-1 \
checkpoint.keep_best_checkpoints=10 \
checkpoint.no_epoch_checkpoints=true \
checkpoint.save_interval_updates=0 \
checkpoint.keep_interval_updates=-1 \
checkpoint.save_dir=${model_path}  \
task.data=/data/zym22/LibriSpeech/manifest/resource  \
task.normalize=true \
dataset.train_subset=train_1h  \
dataset.valid_subset=dev_clean  \
dataset.num_workers=4 \
dataset.max_tokens=3200000  \
dataset.validate_interval=20 \
dataset.validate_after_updates=10000 \
distributed_training.distributed_world_size=2  \
optimization.update_freq=[4] \
optimization.max_update=13000 \
model.w2v_path=/data/zym22/models/multi2vec/multi2vec_baseline/checkpoint700.pt \
model.freeze_finetune_updates=10000 \
common.log_interval=200 \
common.tensorboard_logdir=${model_path}/tensorboard \
common.log_file=${model_path}/log/hydra_train.log \
common.wandb_project=multi2vec \
common.user_dir=multi2vec \
# +criterion.wer_kenlm_model=/data/zym22/models/language_model/4-gram.bin  \
# +criterion.wer_lexicon=/data/zym22/models/language_model/librispeech_lexicon.lst  \
# +criterion.wer_lm_weight=2 \
# +criterion.wer_word_score=-1 \