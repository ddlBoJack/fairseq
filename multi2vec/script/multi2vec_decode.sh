#!/bin/bash
export PYTHONPATH=~/fairseq:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd ~/fairseq

checkpoint_dir=/data/zym22/models/multi2vec/multi2vec_baseline/checkpoint700/train_100h_4-gram
export CUDA_VISIBLE_DEVICES=1

# viterbi
for SPLIT in dev_clean dev_other test_clean test_other; do \
    decode_output_dir=${checkpoint_dir}/${SPLIT}/viterbi
    python examples/speech_recognition/new/infer.py \
    --config-dir examples/speech_recognition/new/conf \
    --config-name infer \
    task=audio_finetuning \
    task.data=/data/zym22/LibriSpeech/manifest/resource \
    task.labels=ltr \
    task.normalize=true \
    dataset.gen_subset=${SPLIT} \
    decoding.type=viterbi  \
    decoding.lmweight=2 decoding.wordscore=-1 decoding.silweight=0 \
    decoding.beam=1500 \
    decoding.unique_wer_file=True \
    common_eval.path=${checkpoint_dir}/checkpoint_best.pt \
    common_eval.results_path=${decode_output_dir} \
    common_eval.quiet=true \
    common.user_dir=multi2vec \
    distributed_training.distributed_world_size=1 \
    hydra.run.dir=${decode_output_dir}
done

# # 4-gram
# for SPLIT in dev_clean dev_other test_clean test_other; do \
#     decode_output_dir=${checkpoint_dir}/${SPLIT}/4-gram
#     python examples/speech_recognition/new/infer.py \
#     --config-dir examples/speech_recognition/new/conf \
#     --config-name infer \
#     task=audio_finetuning \
#     task.data=/data/zym22/LibriSpeech/manifest/resource \
#     task.labels=ltr \
#     task.normalize=true \
#     dataset.gen_subset=${SPLIT} \
#     decoding.type=kenlm  \
#     decoding.lmweight=2 decoding.wordscore=-1 decoding.silweight=0 \
#     decoding.beam=1500 \
#     decoding.unique_wer_file=True \
#     common_eval.path=${checkpoint_dir}/checkpoint_best.pt \
#     common_eval.results_path=${decode_output_dir} \
#     common_eval.quiet=true \
#     common.user_dir=multi2vec \
#     distributed_training.distributed_world_size=1 \
#     hydra.run.dir=${decode_output_dir}  \
#     decoding.lexicon=/data/zym22/models/language_model/librispeech_lexicon.lst \
#     decoding.lmpath=/data/zym22/models/language_model/4-gram.bin
# done

# #from official
# python examples/speech_recognition/new/infer.py 
# --config-dir examples/speech_recognition/new/conf \
# --config-name infer \
# task=audio_finetuning \
# task.data=/path/to/manifests \
# common.user_dir=examples/data2vec \
# task.labels=ltr \
# decoding.type=kenlm \
# decoding.lmweight=${lmweight} \
# decoding.wordscore=${wordscore} \
# decoding.silweight=${silscore} \
# decoding.lexicon=/path/to/lexicon \
# decoding.lmpath=/path/to/lm \
# decoding.unique_wer_file=True \
# dataset.gen_subset=dev_clean,dev_other,test_clean,test_other \
# common_eval.path=/path/to/checkpoint.pt \
# decoding.beam=1500 \
# distributed_training.distributed_world_size=${num_gpus}

# #from yujin
# model=<model_path>
# decode_data_type=<dev-clean|dev-other|test-clean|test-other>

# cd <fairseq_dir>
# python3 examples/speech_recognition/new/infer.py \
#         --config-dir <config_dir> \
#         --config-name <config_name> \
#         task.data=/mnt/lustre/sjtu/home/xc915/superb/dataset/librispeech_finetuning_data/${decode_data_type} \
#         task.normalize=<true|false> \ 
#         common_eval.path=${model} \
#         dataset.gen_subset=test_2 \ # test指向/lusture下的数据，test_2指向/home下的数据，或者根据tsv的名字直接改
#         decoding.lexicon=/mnt/lustre/sjtu/home/xc915/superb/nlp_utils/lexicon/librispeech_lexicon.lst \
#         decoding.lmpath=/mnt/lustre/sjtu/home/xc915/superb/nlp_utils/arpa/4-gram.mmap \
#         hydra.run.dir=<run_dir>