import pandas as pd
import re
import os
import glob
import numpy as np
import torch

data_dir = "../new_data/"

def list_save(filename, data):
    with open(filename, 'w') as file:
        file.writelines(data)
    print(f"{filename} file saved successfully")

def set_save(filename, data):
    with open(filename, 'w') as file:
        file.writelines([line + '\n' for line in data])
    print(f"{filename} file saved successfully")

# 定义算法文件夹
algorithm_dirs = {
    "delly": data_dir + "delly/",
    "manta": data_dir + "manta/",
    "smoove": data_dir + "smoove/",
    "wham": data_dir + "wham/"
}

# 为每个算法分别处理
for algo, algo_dir in algorithm_dirs.items():
    if not os.path.exists(algo_dir):
        print(f"Directory {algo_dir} does not exist, skipping...")
        continue

    # 初始化输出
    insert = ["CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\n"]
    delete = ["CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\n"]
    chr_list = set()
    feature_data = {}  # 存储kmer, softclip, cov特征

    # 获取文件夹下所有.txt文件
    txt_files = glob.glob(os.path.join(algo_dir, "*.txt"))
    print(f"Processing {algo} files: {len(txt_files)} found")

    for filename in txt_files:
        with open(filename, "r") as f:
            lines = f.readlines()
            var_data = None
            call_filter = "PASS"  # 默认PASS
            ac = "0"
            gt_ac = "0"
            bp_count = "0"
            cov_list = []
            kmer_list = []
            softclip_list = []
            insertlen_list = []

            # 解析.txt文件
            for line in lines:
                line = line.strip()
                if line.startswith("@VAR"):
                    # 如果已经有var_data，处理之前的变异
                    if var_data:
                        # 处理INS变异的END和SVLEN
                        if var_data["SVTYPE"] == "INS":
                            svlen = int(np.mean([int(x) for x in insertlen_list])) if insertlen_list else 100
                            var_data["END"] = var_data["POS"] + svlen
                            info = f"SVTYPE=INS;SVLEN={svlen};AC={ac};BP_COUNT={bp_count};COV={','.join(cov_list[:10])};KMER={','.join(kmer_list[:10])};SOFTCLIP={','.join(softclip_list[:10])}"
                        else:
                            svlen = var_data["END"] - var_data["POS"]
                            info = f"SVTYPE=DEL;SVLEN={svlen};AC={ac};BP_COUNT={bp_count};COV={','.join(cov_list[:10])};KMER={','.join(kmer_list[:10])};SOFTCLIP={','.join(softclip_list[:10])}"
                        
                        # 构造VCF格式行
                        vcf_line = f"{var_data['CHROM']}\t{var_data['POS']}\t{var_data['ID']}\tN\tN\t.\t{call_filter}\t{info}\tGT\t{gt_ac}\n"
                        if var_data["SVTYPE"] == "DEL":
                            delete.append(vcf_line)
                        elif var_data["SVTYPE"] == "INS":
                            insert.append(vcf_line)

                        # 保存特征
                        feature_data[var_data["ID"]] = {
                            "kmer": [int(x) for x in kmer_list[:3999]] if kmer_list else [0] * 3999,
                            "softclip": [int(x) for x in softclip_list[:3999]] if softclip_list else [0] * 3999,
                            "cov": [int(x) for x in cov_list[:3999]] if cov_list else [0] * 3999
                        }

                    # 重置变量，处理新的@VAR
                    var_data = None
                    call_filter = "PASS"
                    ac = "0"
                    gt_ac = "0"
                    bp_count = "0"
                    cov_list = []
                    kmer_list = []
                    softclip_list = []
                    insertlen_list = []

                    # 解析新的@VAR
                    parts = line.split(":")
                    if len(parts) < 5:
                        continue
                    _, _, chrom, pos, end, svtype = parts[:6]
                    # chrom = chrom.replace("chr", "")  # 统一染色体命名
                    pos = int(pos)
                    end = int(end) if svtype == "DEL" else pos
                    var_id = f"{algo}_{os.path.basename(filename).split('.')[0]}_{pos}_{svtype}"
                    chr_list.add(chrom)
                    var_data = {
                        "CHROM": chrom,
                        "POS": pos,
                        "END": end,
                        "SVTYPE": svtype,
                        "ID": var_id
                    }
                    # print(f"Processing {algo} {filename}: {var_data['CHROM']}:{var_data['POS']}:{var_data['SVTYPE']}")
                elif line.startswith("@CALL_FILTER"):
                    call_filter = line.split(":")[1]
                elif line.startswith("@AC"):
                    ac = line.split(":")[1]
                elif line.startswith("@GT_AC"):
                    gt_ac = line.split(":")[1]
                elif line.startswith("@BP_COUNT"):
                    bp_count = line.split(":")[1]
                elif line.startswith("@COV"):
                    cov_list = line.split(":")[1].split(",")
                elif line.startswith("@KMER_LIST"):
                    kmer_list = line.split(":")[1].split(",")
                elif line.startswith("@SOFTCLIP"):
                    softclip_list = line.split(":")[1].split(",")
                elif line.startswith("@INSERTLEN"):
                    insertlen_list = line.split(":")[1].split(",")

            # 处理最后一个@VAR
            if var_data:
                if var_data["SVTYPE"] == "INS":
                    svlen = int(np.mean([int(x) for x in insertlen_list])) if insertlen_list else 100
                    var_data["END"] = var_data["POS"] + svlen
                    info = f"SVTYPE=INS;SVLEN={svlen};AC={ac};BP_COUNT={bp_count};COV={','.join(cov_list[:10])};KMER={','.join(kmer_list[:10])};SOFTCLIP={','.join(softclip_list[:10])}"
                else:
                    svlen = var_data["END"] - var_data["POS"]
                    info = f"SVTYPE=DEL;SVLEN={svlen};AC={ac};BP_COUNT={bp_count};COV={','.join(cov_list[:10])};KMER={','.join(kmer_list[:10])};SOFTCLIP={','.join(softclip_list[:10])}"
                
                vcf_line = f"{var_data['CHROM']}\t{var_data['POS']}\t{var_data['ID']}\tN\tN\t.\t{call_filter}\t{info}\tGT\t{gt_ac}\n"
                if var_data["SVTYPE"] == "DEL":
                    delete.append(vcf_line)
                elif var_data["SVTYPE"] == "INS":
                    insert.append(vcf_line)

                feature_data[var_data["ID"]] = {
                    "kmer": [int(x) for x in kmer_list[:3999]] if kmer_list else [0] * 3999,
                    "softclip": [int(x) for x in softclip_list[:3999]] if softclip_list else [0] * 3999,
                    "cov": [int(x) for x in cov_list[:3999]] if cov_list else [0] * 3999
                }

    # 保存算法特定的VCF和染色体列表
    list_save(data_dir + f"{algo}_ins.vcf", insert)
    list_save(data_dir + f"{algo}_del.vcf", delete)
    set_save(data_dir + f"{algo}_chr.txt", chr_list)

    # 保存特征
    os.makedirs(data_dir + f"features/{algo}", exist_ok=True)
    for var_id, features in feature_data.items():
        torch.save(features, data_dir + f"features/{algo}/{var_id}.pt")

    # 转换为CSV-VCF格式
    insert_result_data = pd.read_csv(data_dir + f"{algo}_ins.vcf", sep="\t")
    insert_result_data.insert(2, 'SPOS', 0)  # 无CIPOS，设为0
    insert_result_data.insert(3, 'EPOS', 0)
    insert_result_data.insert(4, 'SVLEN', 0)

    for index, row in insert_result_data.iterrows():
        print(f"{algo} INS index = {index}", end='\r')
        s = row["INFO"]
        pos = s.find("SVLEN")
        if pos != -1:
            pos += 6
            s = s[pos:].split(";")[0]
            insert_result_data.loc[index, "SVLEN"] = int(s)
        insert_result_data.loc[index, "SPOS"] = 0
        insert_result_data.loc[index, "EPOS"] = 0

    insert_result_data.to_csv(data_dir + f"insert_result_data_{algo}.csv.vcf", sep="\t")
    print(f"{algo} INS finished, total number = {index}")

    delete_result_data = pd.read_csv(data_dir + f"{algo}_del.vcf", sep="\t")
    delete_result_data.insert(2, 'SPOS', 0)
    delete_result_data.insert(3, 'EPOS', 0)
    delete_result_data.insert(4, 'END', 0)
    delete_result_data.insert(5, 'SEND', 0)
    delete_result_data.insert(6, 'EEND', 0)

    for index, row in delete_result_data.iterrows():
        print(f"{algo} DEL index = {index}", end='\r')
        s = row["INFO"]
        pos = s.find("SVLEN")
        if pos != -1:
            pos += 6
            s = s[pos:].split(";")[0]
            svlen = int(s)
            delete_result_data.loc[index, "SVLEN"] = svlen
            delete_result_data.loc[index, "END"] = row["POS"] + abs(svlen)
        delete_result_data.loc[index, "SPOS"] = 0
        delete_result_data.loc[index, "EPOS"] = 0
        delete_result_data.loc[index, "SEND"] = 0
        delete_result_data.loc[index, "EEND"] = 0

    delete_result_data.to_csv(data_dir + f"delete_result_data_{algo}.csv.vcf", sep="\t")
    print(f"{algo} DEL finished, total number = {index}")