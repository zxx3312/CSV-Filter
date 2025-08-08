from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from scipy import interp
from sklearn.metrics import auc, classification_report, roc_curve
import torch
from net import IDENet
import sys
import os

# 设置随机种子
seed_everything(2022)

# 数据和模型根目录
root_dir = "../"
data_dir = root_dir + "data/"

# 配置
config = {
    "lr": 7.1873e-06,
    "batch_size": 118,
    "beta1": 0.9,
    "beta2": 0.999,
    'weight_decay': 0.0011615,
    'model': None  # 将在循环中设置
}

# 算法列表
algorithms = ["delly", "manta", "smoove", "wham"]

# 为每个算法生成预测结果
for algo in algorithms:
    print(f"Processing algorithm: {algo}")

    # 更新配置中的模型名称
    config['model'] = algo

    # 检查模型文件是否存在
    model_path = root_dir + f"models/{algo}.ckpt"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist, skipping {algo}...")
        continue

    # 加载模型
    model = IDENet.load_from_checkpoint(
        model_path, path=data_dir + f"image/{algo}/", config=config
    )

    # 初始化训练器
    trainer = pl.Trainer(gpus=1)
    model.eval()

    # 测试模型
    result = trainer.test(model)

    # 加载测试结果
    result_path = data_dir + f"result_{algo}.pt"
    if not os.path.exists(result_path):
        print(f"Result file {result_path} does not exist, skipping {algo}...")
        continue
    output = torch.load(result_path)

    # 初始化y和y_hat
    y = torch.empty(0, 3)
    y_hat = torch.empty(0, 3).cuda()

    # 处理输出，生成标签和预测概率
    for out in output:
        for ii in out['y']:
            if ii == 0:
                y = torch.cat([y, torch.tensor([1, 0, 0]).unsqueeze(0)], 0)
            elif ii == 1:
                y = torch.cat([y, torch.tensor([0, 1, 0]).unsqueeze(0)], 0)
            else:
                y = torch.cat([y, torch.tensor([0, 0, 1]).unsqueeze(0)], 0)
        y_hat = torch.cat([y_hat, out['y_hat']], 0)

    # 转换为numpy数组
    y_test = y.cpu().numpy()
    y_score = y_hat.cpu().numpy()
    n_classes = y.shape[1]

    # 打印调试信息
    print(f"{algo} num_elements of y = {y.numel()}")
    print(f"{algo} num_elements of y_hat = {y_hat.numel()}")
    print(f"{algo} type(y) = {type(y)}, type(y_hat) = {type(y_hat)}")
    print(f"{algo} type(y_test) = {type(y_test)}, type(y_score) = {type(y_score)}, type(n_classes) = {type(n_classes)}")
    print(f"{algo} size of y = {sys.getsizeof(y)}, size of y_hat = {sys.getsizeof(y_hat)}")
    print(f"{algo} size of y_test = {sys.getsizeof(y_test)}, size of y_score = {sys.getsizeof(y_score)}, size of n_classes = {sys.getsizeof(n_classes)}")

    # 计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘制ROC曲线
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'{algo} micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'{algo} macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'{algo} ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f"roc_{algo}.pdf", dpi=1000, bbox_inches='tight')
    plt.close()

    # 输出分类报告
    print(f"\nClassification Report for {algo}:")
    print(classification_report(torch.argmax(y.cpu(), dim=1),
                                torch.argmax(y_hat.cpu(), dim=1), digits=4))