#!/usr/bin/env python3 -u
# Author: Ziyang Ma
# Create Date: 2023-02-16
from __future__ import print_function

import argparse
import logging
import time
import sys
import os
import subprocess
import torch

# os.environ["NCCL_DEBUG_SUBSYS"]="ALL"
# os.environ["NCCL_DEBUG"]="INFO"

sys.path.append("./")

if __name__ == '__main__':

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print("device count %d" % (device_count))
    else:
        print("could not find the gpu device")

    # # multi-process read lmdb
    # import lmdb
    # import multiprocessing as mp
    # from multiprocessing import Pool
    # from functools import partial

    # # global lmdb_dict, which can be used in fairseq train.py
    # trian_960_wav = mp.Manager().dict()

    # def read_lmdb(lmdb_path, lmdb_dict):
    #     lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    #     lmdb_txn = lmdb_env.begin(write=False)
    #     lmdb_cursor = lmdb_txn.cursor()
    #     for key, value in lmdb_cursor:
    #         lmdb_dict[key] = value
    #     lmdb_env.close()

    # lmdb_path = "/data/volume1/maziyang.mzy/LibriSpeech/pai_lmdb/train_960_parallel"
    # lmdb_path_list = [lmdb_path + "/train_960_{}".format(i) for i in range(32)]
    # pool = Pool(processes=32)
    # pool.map(partial(read_lmdb, lmdb_dict=trian_960_wav), lmdb_path_list)

    
    command = ". submit_script/pretrain/hubert_modified/hubert_modified_pretrain.sh"
    print(command)
    subprocess.call(command, shell=True)