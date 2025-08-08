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

# 数据目录
bam_data_dir = "../new_data/"
vcf_data_dir = "../new_data/"
data_dir = "../new_data/"

# BAM 文件路径
bam_path = bam_data_dir + "HG00096.sorted.bam"

# 算法列表
algos = ["manta", "smoove", "wham"]

# 读取 BAM 文件中的染色体信息
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list_sam_file = sam_file.references
chr_length_sam_file = sam_file.lengths
sam_file.close()

# 允许的染色体列表
allowed_chromosomes = set(f"chr{i}" for i in range(1, 23)) | {"chrX", "chrY"}

chr_list = []
chr_length = []

# 筛选允许的染色体
for chrom, length in zip(chr_list_sam_file, chr_length_sam_file):
    if chrom in allowed_chromosomes:
        chr_list.append(chrom)
        chr_length.append(length)

# 遍历每种算法
for algo in algos:
    print(f"======= Checking algo: {algo} =======")
    
    # VCF 文件路径，动态包含 algo
    ins_vcf_filename = vcf_data_dir + f"insert_result_data_{algo}.csv.vcf"
    del_vcf_filename = vcf_data_dir + f"delete_result_data_{algo}.csv.vcf"
    
    # 检查 VCF 文件是否存在
    if not os.path.exists(ins_vcf_filename):
        print(f"Error: VCF file {ins_vcf_filename} not found.")
        exit(1)
    if not os.path.exists(del_vcf_filename):
        print(f"Error: VCF file {del_vcf_filename} not found.")
        exit(1)
    
    # 检查每个染色体的 negative_cigar_new_img.pt 文件
    for chromosome, chr_len in zip(chr_list, chr_length):
        print(f"======= Check {chromosome} for algo {algo} =======")
        file_path = data_dir + f'image/{algo}/{chromosome}/negative_cigar_new_img.pt'
        if os.path.exists(file_path):
            print(f"{file_path} exists.")
        else:
            print(f'Check fail! File {file_path} not found. Reduce the parameters and re-run the "python parallel_process_file.py --thread_num thread_num --algo {algo}" command!')
            exit(1)
    
    print(f'Check success for algo {algo}!')

print("All algorithms checked successfully!")