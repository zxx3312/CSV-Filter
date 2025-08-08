import torch

output = torch.load("./result.pt", map_location="cpu")

print(type(output))  # 应该是 list
print(len(output))
print(output[0].keys())  # 应该有 'y', 'y_hat'
print(output[0]['y'])    # 标签
print(output[0]['y_hat'].shape)  # 模型预测
