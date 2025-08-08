import utilities as ut
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
import os
from net import IDENet
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import Pool, cpu_count
import pysam
import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import list2img
from hyperopt import hp

# bam_data_dir = "../new_data/"
bam_data_dir = "../new_data/"
vcf_data_dir = "../new_data/"
data_dir = "../new_data/"
bam_path = bam_data_dir + "test.bam"
# bam_path = bam_data_dir + "HG002_PacBio_GRCh38.bam"
bam_path = bam_data_dir + "HG00096.sorted.bam"

ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"

sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list_sam_file = sam_file.references
chr_length_sam_file = sam_file.lengths
sam_file.close()

# allowed_chromosomes = set(f"{i}" for i in range(1, 23)) | {"X", "Y"}
allowed_chromosomes = set(f"chr{i}" for i in range(1, 23)) | {"chrX", "chrY"}

chr_list = []
chr_length = []

for chrom, length in zip(chr_list_sam_file, chr_length_sam_file):
    if chrom in allowed_chromosomes:
        chr_list.append(chrom)
        # chr_list.append(chrom[3:] if chrom.startswith("chr") else chrom)
        chr_length.append(length)


for chromosome, chr_len in zip(chr_list, chr_length):
    print("======= check " + chromosome + " =======")
    if os.path.exists(data_dir + 'image/' + chromosome + '/negative_cigar_new_img' + '.pt'):
        pass
    else:
        print('Check fail! Reduce the parameters and re-run the "python parallel_process_file.py --thread_num thread_num" command!')
        exit(1)
print('Check success!')
