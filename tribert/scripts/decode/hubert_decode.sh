#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

checkpoint_dir=/mnt/lustre/sjtu/home/zym22/models/tribert/u_triphone1744_hmmdmm/checkpoint_last/train_100h
export CUDA_VISIBLE_DEVICES=0,1,2,3

# infer_viterbi
for SPLIT in dev_clean dev_other test_clean test_other; do \
    decode_output_dir=${checkpoint_dir}/viterbi/${SPLIT}
    # python -m debugpy --listen 5678 --wait-for-client examples/speech_recognition/new/infer.py  \
    python examples/speech_recognition/new/infer.py \
    --config-dir /mnt/lustre/sjtu/home/zym22/fairseq/examples/hubert/config/decode \
    --config-name infer_viterbi \
    task.data=/mnt/lustre/sjtu/home/zym22/data/manifest/resource \
    task.normalize=false \
    dataset.gen_subset=${SPLIT} \
    decoding.lmweight=2 decoding.wordscore=-1 decoding.silweight=0 \
    decoding.beam=1500 \
    common_eval.path=${checkpoint_dir}/checkpoint_best.pt \
    common_eval.results_path=${decode_output_dir} \
    common_eval.quiet=true \
    distributed_training.distributed_world_size=4 \
    hydra.run.dir=${decode_output_dir} \
    #common.user_dir=multi2vec
done

# # infer_kenlm
# for SPLIT in dev_clean dev_other test_clean test_other; do \
#     decode_output_dir=${checkpoint_dir}/4-gram/${SPLIT}
#     # python -m debugpy --listen 5678 --wait-for-client examples/speech_recognition/new/infer.py  \
#     python examples/speech_recognition/new/infer.py \
#     --config-dir /mnt/lustre/sjtu/home/zym22/fairseq/examples/hubert/config/decode \
#     --config-name infer_kenlm \
#     task.data=/data/zym22/LibriSpeech/manifest/resource \
#     task.normalize=false \
#     dataset.gen_subset=${SPLIT} \
#     decoding.lmweight=2 decoding.wordscore=-1 decoding.silweight=0 \
#     decoding.beam=1500 \
#     common_eval.path=${checkpoint_dir}/checkpoint_best.pt \
#     common_eval.results_path=${decode_output_dir} \
#     common_eval.quiet=true \
#     distributed_training.distributed_world_size=1 \
#     hydra.run.dir=${decode_output_dir} \
#     decoding.lexicon=/data/zym22/models/language_model/librispeech_lexicon.lst \
#     decoding.lmpath=/data/zym22/models/language_model/4-gram.bin
#     #common.user_dir=multi2vec
# done