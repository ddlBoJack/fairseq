#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

# edit your exp
model_name=multi2vec
exp_name=hubert_baseline
checkpoint=checkpoint700
finetune=train_1h
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
--config-name hubert_base_10h  \
checkpoint.save_dir=${model_path}  \
checkpoint.save_interval=50 \
checkpoint.save_interval_updates=1000 \
task.data=/data/zym22/LibriSpeech/manifest/resource  \
task.label_dir=/data/zym22/LibriSpeech/manifest/resource \
task.normalize=false \
dataset.skip_invalid_size_inputs_valid_test=true \
dataset.validate_interval=1000 \
dataset.train_subset=train_1h  \
dataset.valid_subset=dev_other  \
distributed_training.distributed_world_size=2  \
optimization.update_freq=[4] \
optimization.max_update=13000 \
model.w2v_path=/data/zym22/models/multi2vec/hubert_baseline/checkpoint700.pt \
common.wandb_project=multi2vec \
# +criterion.wer_kenlm_model=${kenlm_model_path}  \
# +criterion.wer_lexicon=${lexicon_path}  \
# +criterion.wer_lm_weight=2 \
# +criterion.wer_word_score=-1 \