import subprocess
import os
from utilities import mymkdir
from tqdm import tqdm
from collections import defaultdict
from multiprocessing.dummy import Pool  # 用线程代替进程
from multiprocessing import cpu_count



# bam_name = "HG002-ONT-minimap2.sorted.bam"
# bam_name = "HG002_PacBio_GRCh38.bam"
bam_name = "HG00096.sorted.bam"
output_file = "output.depth.txt"
# bam_data_dir = "../new_data/"
bam_data_dir = "../new_data/"
# data_dir = "../new_data/"
data_dir = "../new_data/"

cmd = "samtools depth " + bam_data_dir + bam_name + " > " + data_dir + output_file
print(cmd)
print("==== starting samtools deal ====")
# subprocess.call(cmd, shell = True)

mymkdir(data_dir + "depth/")

# print("==== starting process depth file ====")
# with open(data_dir + output_file, "r") as f:
#     lines = f.readlines()  
#     for line in tqdm(lines, desc="Processing lines", unit="line"):
#         with open(data_dir + "depth/" + line.split("\t")[0], "a+") as subf:
#             subf.write(line)

# 构建染色体 -> 所有 depth line 的映射
chrom_lines = defaultdict(list)
print("==== loading depth file ====")
with open(data_dir + output_file, "r") as f:
    for line in tqdm(f, desc="Loading depth file", unit="line"):
        chrom = line.split("\t")[0]
        chrom_lines[chrom].append(line)

print("==== start multiprocessing write ====")

def write_chromosome_depth(args):
    print("in write_chromosome_depth")
    chrom, lines = args
    out_path = os.path.join(data_dir, "depth", chrom)
    with open(out_path, "w") as f:
        f.writelines(lines)
    return chrom

with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(write_chromosome_depth, chrom_lines.items()), total=len(chrom_lines)))

print("Done! Processed chromosomes:", results)
            
