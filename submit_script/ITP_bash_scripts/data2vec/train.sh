#!/bin/bash

lr=0.0003
dropout=0.1
size=25
seed=1

set -e
source ~/miniconda/etc/profile.d/conda.sh
echo 'Activate nat env...'
conda activate nat
cd nat
python train.py /mnt/exp/data --arch nat \
    --noise nat_mask --share-all-embeddings --criterion nat_loss --label-smoothing 0.1 \
    --lr ${lr} --warmup-init-lr 1e-7 --stop-min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_lev_modified --max-tokens 2048 \
    --weight-decay 0.01 --dropout ${dropout} --encoder-layers 12 --encoder-embed-dim 512 --decoder-layers 2 \
    --decoder-embed-dim 512 --fp16 --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 \
    --seed ${seed} --clip-norm 5 --save-dir /mnt/exp/project \
    --src-embedding-copy --log-interval 1000 --user-dir block_plugins --block-size ${size} --total-up 300000 \
    --update-freq 16 --decoder-learned-pos --encoder-learned-pos --apply-bert-init --activation-fn gelu
cd scripts
python average_checkpoints.py \
    --inputs /mnt/exp/project/NMT \
    --num-epoch-checkpoints 10 \
    --output /mnt/exp/project/NMT
echo -e '\n'
exec bash