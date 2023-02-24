#!/usr/bin/env python3 -u
# Author: Ziyang Ma
# Create Date: 2023-02-16
from __future__ import print_function

import argparse
import copy
import logging
import time
import copy
import sys
import subprocess
import os

os.environ["NCCL_DEBUG_SUBSYS"]="ALL"
os.environ["NCCL_DEBUG"]="INFO"

sys.path.append("./")

from multiprocessing import Pool

if __name__ == '__main__':

    command = ". submit_script/pretrain/data2vec/data2vec_pretrain.sh"
    print(command)
    subprocess.call(command, shell=True)