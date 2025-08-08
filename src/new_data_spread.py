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

# 强制刷新标志
all_enforcement_refresh = 0
position_enforcement_refresh = 0
img_enforcement_refresh = 0
sign_enforcement_refresh = 0 
cigar_enforcement_refresh = 0

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

hight = 224

# 遍历每种算法
for algo in algos:
    print(f"======= Processing algo: {algo} =======")
    
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

    # 检查是否需要重新生成 all_*_img.pt 文件
    if os.path.exists(data_dir + f'/all_n_img_{algo}.pt') and not all_enforcement_refresh:
        print(f"Loading existing all_*_img_{algo}.pt files")
        all_ins_cigar_img = torch.load(data_dir + f'/all_ins_img_{algo}.pt')
        all_del_cigar_img = torch.load(data_dir + f'/all_del_img_{algo}.pt')
        all_negative_cigar_img = torch.load(data_dir + f'/all_n_img_{algo}.pt')
    else:
        all_ins_cigar_img = torch.empty(0, 1, hight, hight)
        all_del_cigar_img = torch.empty(0, 1, hight, hight)
        all_negative_cigar_img = torch.empty(0, 1, hight, hight)

        for chromosome, chr_len in zip(chr_list, chr_length):
            print(f"======= Deal {chromosome} for algo {algo} =======")

            # 处理位置信息
            print("position start")
            position_path = data_dir + f'position/{algo}/{chromosome}'
            if os.path.exists(position_path + '/negative.pt') and not position_enforcement_refresh:
                print("loading position files")
                ins_position = torch.load(position_path + '/insert.pt')
                del_position = torch.load(position_path + '/delete.pt')
                n_position = torch.load(position_path + '/negative.pt')
            else:
                ins_position = []
                del_position = []
                n_position = []
                # 读取 VCF 文件（移除 index_col=0）
                insert_result_data = pd.read_csv(ins_vcf_filename, sep="\t", comment="#")
                delete_result_data = pd.read_csv(del_vcf_filename, sep="\t", comment="#")
                
                # 筛选染色体
                insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
                row_pos = []
                for index, row in insert_chromosome.iterrows():
                    row_pos.append(row["POS"])

                set_pos = set()
                for pos in row_pos:
                    set_pos.update(range(pos - 100, pos + 100))

                for pos in row_pos:
                    gap = 112
                    begin = pos - 1 - gap
                    end = pos - 1 + gap
                    if begin < 0:
                        begin = 0
                    if end >= chr_len:
                        end = chr_len - 1
                    ins_position.append([begin, end])

                # 删除位点
                delete_chromosome = delete_result_data[delete_result_data["CHROM"] == chromosome]
                row_pos = []
                row_end = []
                for index, row in delete_chromosome.iterrows():
                    row_pos.append(row["POS"])
                    row_end.append(row["END"])

                for pos in row_pos:
                    set_pos.update(range(pos - 100, pos + 100))

                for pos, end in zip(row_pos, row_end):
                    gap = int((end - pos) / 4) or 1
                    begin = pos - 1 - gap
                    end = end - 1 + gap
                    if begin < 0:
                        begin = 0
                    if end >= chr_len:
                        end = chr_len - 1
                    del_position.append([begin, end])

                    # 负样本
                    del_length = end - begin
                    for _ in range(2):
                        end = begin
                        while end - begin < del_length / 2 + 1:
                            random_begin = random.randint(1, chr_len)
                            while random_begin in set_pos:
                                random_begin = random.randint(1, chr_len)
                            begin = random_begin - 1 - gap
                            end = begin + del_length
                            if begin < 0:
                                begin = 0
                            if end >= chr_len:
                                end = chr_len - 1
                        n_position.append([begin, end])

                # 保存位置文件
                ut.mymkdir(position_path)
                torch.save(ins_position, position_path + '/insert.pt')
                torch.save(del_position, position_path + '/delete.pt')
                torch.save(n_position, position_path + '/negative.pt')
            print("position end")

            # 生成图像
            print("cigar start")
            image_path = data_dir + f'image/{algo}/{chromosome}'
            if os.path.exists(image_path + '/negative_cigar_new_img.pt') and not cigar_enforcement_refresh:
                print("loading image files")
                ins_cigar_img = torch.load(image_path + '/ins_cigar_new_img.pt')
                del_cigar_img = torch.load(image_path + '/del_cigar_new_img.pt')
                negative_cigar_img = torch.load(image_path + '/negative_cigar_new_img.pt')
            else:
                ins_cigar_img = torch.empty(len(ins_position), 1, hight, hight)
                del_cigar_img = torch.empty(len(del_position), 1, hight, hight)
                negative_cigar_img = torch.empty(len(n_position), 1, hight, hight)
                
                for i, b_e in enumerate(ins_position):
                    zoom = 1
                    fail = 1
                    while fail:
                        try:
                            fail = 0
                            ins_cigar_img[i] = ut.cigar_new_img_single_optimal(
                                bam_path, chromosome, b_e[0], b_e[1], zoom)
                        except Exception as e:
                            fail = 1
                            zoom += 1
                            print(e)
                            print(f"Exception cigar_img_single_optimal (ins) {algo} {chromosome} zoom={zoom}")
                    print(f"===== finish(ins_cigar_img) {algo} {chromosome} {i}/{len(ins_position)}")

                for i, b_e in enumerate(del_position):
                    zoom = 1
                    fail = 1
                    while fail:
                        try:
                            fail = 0
                            del_cigar_img[i] = ut.cigar_new_img_single_optimal(
                                bam_path, chromosome, b_e[0], b_e[1], zoom)
                        except Exception as e:
                            fail = 1
                            zoom += 1
                            print(e)
                            print(f"Exception cigar_img_single_optimal (del) {algo} {chromosome} zoom={zoom}")
                    print(f"===== finish(del_position) {algo} {chromosome} {i}/{len(del_position)}")

                for i, b_e in enumerate(n_position):
                    zoom = 1
                    fail = 1
                    while fail:
                        try:
                            fail = 0
                            negative_cigar_img[i] = ut.cigar_new_img_single_optimal(
                                bam_path, chromosome, b_e[0], b_e[1], zoom)
                        except Exception as e:
                            fail = 1
                            zoom += 1
                            print(e)
                            print(f"Exception cigar_img_single_optimal (negative) {algo} {chromosome} zoom={zoom}")
                    print(f"===== finish(n_position) {algo} {chromosome} {i}/{len(n_position)}")

                # 保存图像文件
                ut.mymkdir(image_path)
                torch.save(ins_cigar_img, image_path + '/ins_cigar_new_img.pt')
                torch.save(del_cigar_img, image_path + '/del_cigar_new_img.pt')
                torch.save(negative_cigar_img, image_path + '/negative_cigar_new_img.pt')
            print("cigar end")

            # 合并图像
            all_ins_cigar_img = torch.cat((all_ins_cigar_img, ins_cigar_img), 0)
            all_del_cigar_img = torch.cat((all_del_cigar_img, del_cigar_img), 0)
            all_negative_cigar_img = torch.cat((all_negative_cigar_img, negative_cigar_img), 0)

        # 保存合并后的图像
        torch.save(all_ins_cigar_img, data_dir + f'/all_ins_img_{algo}.pt')
        torch.save(all_del_cigar_img, data_dir + f'/all_del_img_{algo}.pt')
        torch.save(all_negative_cigar_img, data_dir + f'/all_n_img_{algo}.pt')

    print(f"Loading data for algo {algo}")
    all_ins_img = torch.load(data_dir + f'/all_ins_img_{algo}.pt')
    all_del_img = torch.load(data_dir + f'/all_del_img_{algo}.pt')
    all_n_img = torch.load(data_dir + f'/all_n_img_{algo}.pt')

    length = len(all_ins_img) + len(all_del_img) + len(all_n_img)
    print(f"Total images for algo {algo}: {length}")

    # 创建保存目录
    ut.mymkdir(data_dir + f"ins/{algo}")
    ut.mymkdir(data_dir + f"del/{algo}")
    ut.mymkdir(data_dir + f"n/{algo}")

    # 保存单独的图像文件
    for index in range(length):
        print(f"Loaded index = {index}/{length} for algo {algo}", end='\r')
        if index < len(all_ins_img):
            image = all_ins_img[index].clone()
            torch.save([image, 2], data_dir + f"ins/{algo}/{index}.pt")
        elif index < len(all_ins_img) + len(all_del_img):
            index -= len(all_ins_img)
            image = all_del_img[index].clone()
            torch.save([image, 1], data_dir + f"del/{algo}/{index}.pt")
        else:
            index -= len(all_ins_img) + len(all_del_img)
            image = all_n_img[index].clone()
            torch.save([image, 0], data_dir + f"n/{algo}/{index}.pt")

    print(f"All images loaded for algo {algo}, number = {length}")

print("All algorithms processed successfully!")