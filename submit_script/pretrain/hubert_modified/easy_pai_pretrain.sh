#!/bin/bash
# Run pre-train HuBERT on PAI
# Author: Ziyang Ma
# Create Date: 2023-02-11

set -x
set -e
set -u

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=INFO

rm -rf /home/maziyang.mzy/fairseq_mzy
mkdir -p /home/maziyang.mzy/fairseq_mzy
# cp -r /mnt/maziyang.mzy/code/fairseq /home/maziyang.mzy/fairseq_mzy
ls /home/maziyang.mzy/fairseq-pai/ | grep -v None | xargs -i cp -r /home/maziyang.mzy/fairseq-pai/{} /home/maziyang.mzy/fairseq_mzy/
cd /home/maziyang.mzy/fairseq_mzy

cp requirements.pai requirements.txt
odps_project="speech_product_training"
#mode="train"
odpscmd="/home/maziyang.mzy/asrp/src/odpsclt/bin/odpscmd"
volume_project="speech_model_training"
outputs="odps://${volume_project}/volumes/gongzuo/"
volumes="odps://${volume_project}/volumes/gongzuo/"

## some settings
count=1
gpu_num=8

## training config
world_size=`expr $count \* $gpu_num`
#batch_size=32
## prepare the scripts
cur_path=`pwd`
rm -f fairseq.tar.gz
tar --exclude='*.tar.gz' -zcf fairseq.tar.gz *
job_path='file://'${cur_path}'/fairseq.tar.gz'


command="
use speech_product_training;
set odps.algo.hybrid.deploy.info=LABEL:V100M32:OPER_EQUAL;
pai -name=pytorch180
-Dscript=${job_path}
-Dpython=3.6
-DenableDockerFusion=false
-DworkerCount=$count
-DentryFile='-m easypai.torch.launch submit_script/pretrain/hubert_modified/hubert_modified_pretrain.py'
-Doutputs=$outputs
-Dvolumes=$volumes
-Dcluster='{\"worker\":{\"count\":${count},\"gpu\":$[${gpu_num}*100]}}'
;"

###
echo "${command}"
${odpscmd} --project ${odps_project} -e "${command}"
echo "Finish."

