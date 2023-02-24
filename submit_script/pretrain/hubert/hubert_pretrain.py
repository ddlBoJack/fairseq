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

os.environ["NCCL_DEBUG_SUBSYS"]="ALL"
os.environ["NCCL_DEBUG"]="INFO"

sys.path.append("./")

if __name__ == '__main__':

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print("device count %d" % (device_count))
    else:
        print("could not find the gpu device")
    
    command = ". submit_script/pretrain/hubert/hubert_pretrain.sh"
    print(command)
    subprocess.call(command, shell=True)