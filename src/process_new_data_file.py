import argparse
import os
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import seed_everything
import utilities as ut

torch.multiprocessing.set_sharing_strategy('file_system')
seed_everything(2022)

bam_data_dir = "../new_data/"
vcf_data_dir = "../new_data/"
data_dir = "../new_data/"

bam_path = bam_data_dir + "HG00096.sorted.bam"
hight = 224
resize = torchvision.transforms.Resize([hight, hight])

def position(sum_data, algo):
    chromosome, chr_len = sum_data
    # chromosome = chromosome.replace("chr", "")
    ins_position = []
    del_position = []
    n_position = []
    
    try:
        insert_result_data = pd.read_csv(vcf_data_dir + f"insert_result_data_{algo}.csv.vcf", sep="\t", index_col=0)
    except Exception as e:
        print(f"Error reading insert_result_data_{algo}.csv.vcf: {e}")
        insert_result_data = pd.DataFrame()
        return

    try:
        delete_result_data = pd.read_csv(vcf_data_dir + f"delete_result_data_{algo}.csv.vcf", sep="\t", index_col=0)
    except Exception as e:
        print(f"Error reading delete_result_data_{algo}.csv.vcf: {e}")
        delete_result_data = pd.DataFrame()
        return
    
    # # 调试：打印列名和CHROM列内容
    # print(f"insert_result_data columns: {insert_result_data.columns}")
    # print(f"insert_result_data['CHROM'].unique(): {insert_result_data['CHROM'].unique()}")
    # print(f"Comparing CHROM with chromosome: {chromosome}")

    insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
    row_pos = []
    for index, row in insert_chromosome.iterrows():
        row_pos.append(row["POS"])
    # print("compare:", insert_result_data["CHROM"], chromosome)
    # print("list:", list(insert_chromosome.iterrows()))

    set_pos = set()
    for pos in row_pos:
        set_pos.update(range(pos - 1024, pos + 1024))

    for pos in row_pos:
        gap = 1024
        begin = pos - 1 - gap
        end = pos - 1 + gap
        if begin < 0:
            begin = 0
        if end >= chr_len:
            end = chr_len - 1
        ins_position.append([begin, end])

    delete_chromosome = delete_result_data[delete_result_data["CHROM"] == chromosome]
    row_pos = []
    row_end = []
    for index, row in delete_chromosome.iterrows():
        row_pos.append(row["POS"])
        row_end.append(row["END"])

    for pos, end in zip(row_pos, row_end):
        gap = int((end - pos) / 4) or 1
        begin = pos - 1 - gap
        end = end - 1 + gap
        if begin < 0:
            begin = 0
        if end >= chr_len:
            end = chr_len - 1
        del_position.append([begin, end])
        del_length = end - begin
        for _ in range(2):
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

    save_path = data_dir + f'position/{algo}/{chromosome}'
    print(len(ins_position), len(del_position), len(n_position))
    if len(ins_position) == 0 and len(del_position) == 0 and len(n_position) == 0:
        print(algo, chromosome, "no position")
    ut.mymkdir(save_path)
    torch.save(ins_position, save_path + '/insert.pt')
    torch.save(del_position, save_path + '/delete.pt')
    torch.save(n_position, save_path + '/negative.pt')

def create_image(sum_data, algo):
    chromosome, chr_len = sum_data
    print(f"deal {algo} {chromosome}")

    ins_position = torch.load(data_dir + f'position/{algo}/{chromosome}/insert.pt')
    del_position = torch.load(data_dir + f'position/{algo}/{chromosome}/delete.pt')
    n_position = torch.load(data_dir + f'position/{algo}/{chromosome}/negative.pt')

    print(f"{algo} {chromosome} cigar start")
    save_path = data_dir + f'image/{algo}/{chromosome}'
    ut.mymkdir(save_path)

    # if os.path.exists(save_path + '/negative_cigar_new_img.pt'):
    #     return

    ins_cigar_img = torch.empty(len(ins_position), 7, hight, hight)
    del_cigar_img = torch.empty(len(del_position), 7, hight, hight)
    negative_cigar_img = torch.empty(len(n_position), 7, hight, hight)

    # 加载特征
    insert_result_data = pd.read_csv(data_dir + f"insert_result_data_{algo}.csv.vcf", sep="\t")
    delete_result_data = pd.read_csv(data_dir + f"delete_result_data_{algo}.csv.vcf", sep="\t")

    # 处理插入变异（ins_position）
    for i, b_e in enumerate(ins_position):
        zoom = 1
        fail = 1
        var_id = insert_result_data.iloc[i]["ID"]
        features = torch.load(data_dir + f"features/{algo}/{var_id}.pt") if os.path.exists(data_dir + f"features/{algo}/{var_id}.pt") else None
        if b_e[0] >= b_e[1]:
            print(f"⚠️ Skip invalid region (begin >= end): {chromosome}:{b_e[0]}-{b_e[1]}")
            continue
        while fail:
            try:
                fail = 0
                ins_cigar_img[i] = ut.cigar_new_img_single_optimal(bam_path, chromosome, b_e[0], b_e[1], zoom)#, features)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
                print(f"Exception cigar_img_single_optimal(ins_position) {algo} {chromosome} zoom={zoom}. Length={b_e[1] - b_e[0]}")
        print(f"===== finish_cigar_img(ins_position) {algo} {chromosome} index = {i}/{len(ins_position)}")

    # 处理缺失变异（del_position）
    for i, b_e in enumerate(del_position):
        zoom = 1
        fail = 1
        var_id = delete_result_data.iloc[i]["ID"]
        features = torch.load(data_dir + f"features/{algo}/{var_id}.pt") if os.path.exists(data_dir + f"features/{algo}/{var_id}.pt") else None
        if b_e[0] >= b_e[1]:
            print(f"⚠️ Skip invalid region (begin >= end): {chromosome}:{b_e[0]}-{b_e[1]}")
            continue
        while fail:
            try:
                fail = 0
                del_cigar_img[i] = ut.cigar_new_img_single_optimal(bam_path, chromosome, b_e[0], b_e[1], zoom)#, features)
            except Exception as e:
                fail = 1
                zoom += 1
                print(f"Exception cigar_img_single_optimal(del_position) {algo} {chromosome} zoom={zoom}. Length={b_e[1] - b_e[0]}")
        print(f"===== finish_cigar_img(del_position) {algo} {chromosome} index = {i}/{len(del_position)}")

    # 处理负样本（n_position）
    for i, b_e in enumerate(n_position):
        zoom = 1
        fail = 1
        # 负样本没有对应的var_id，使用默认特征（None）
        features = None
        if b_e[0] >= b_e[1]:
            print(f"⚠️ Skip invalid region (begin >= end): {chromosome}:{b_e[0]}-{b_e[1]}")
            continue
        while fail:
            try:
                fail = 0
                negative_cigar_img[i] = ut.cigar_new_img_single_optimal(bam_path, chromosome, b_e[0], b_e[1], zoom)#, features)
            except Exception as e:
                fail = 1
                zoom += 1
                print(f"Exception cigar_img_single_optimal(n_position) {algo} {chromosome} zoom={zoom}. Length={b_e[1] - b_e[0]}")
        print(f"===== finish_cigar_img(n_position) {algo} {chromosome} index = {i}/{len(n_position)}")

    torch.save(ins_cigar_img, save_path + '/ins_cigar_new_img.pt')
    torch.save(del_cigar_img, save_path + '/del_cigar_new_img.pt')
    torch.save(negative_cigar_img, save_path + '/negative_cigar_new_img.pt')
    print(f"{algo} {chromosome} cigar end")

def parse_args():
    parser = argparse.ArgumentParser(description="Process chromosome data")
    parser.add_argument('--chr', help="Chromosome name")
    parser.add_argument('--len', help="Chromosome length")
    parser.add_argument('--algo', help="Algorithm name (delly, manta, smoove, wham)")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    algo = args.algo
    position([args.chr, int(args.len)], algo)
    create_image([args.chr, int(args.len)], algo)