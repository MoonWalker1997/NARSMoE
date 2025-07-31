# 专家的数量不发生变化
#
# 我们可以暂时取消NARS的功能，设置一个维护矩阵的函数，输入是按照顺序的所有专家的激活值
# 这个函数是O(n^2)复杂度的函数，一个一个考察“专家A是一种专家B”的可能性
# 具体来说，A如果和B同时激活，那么就有可能A是一种B或者B是一种A，我们就将这种猜想记录在矩阵里
# 这是拿到专家激活值之后的“第一处理路径”，但是它还不会直接覆盖之前的矩阵，根据这个我们来决定
# “第二处理路径”是使用之前的矩阵按照当前输入进行推理，也就是执行n次矩阵乘法，整体复杂度是O(n^3)，这个可以用GPU加速，但是复杂度摆在这，这只能算是权宜之计
# 第二处理路径是对NARS推理的模拟，我们能够根据矩阵中的关系对当前的专家激活进行衍生判断，如果推理后的激活值和推理前的有冲突，那么就说明专家的关系网络存在缺陷【训练专家】
#
# 在“第一处理路径”上，因为这是一种权宜方法，我们不会记录从开始到结束的所有专家隶属度关系，而是只记录最近k次的隶属度情况

import random

import numpy as np
import torch
from matplotlib import pyplot as plt


class NARSSimulator:
    def __init__(self, num_expert, k):
        self.num_expert = num_expert
        self.k = k

        # Python 列表存储历史关系（CPU上）
        self.matrix = [
            [[random.choice([0, 1])] for _ in range(num_expert)]
            for _ in range(num_expert)
        ]

        self.positive_threshold = 0.5
        self.negative_threshold = 0.5

    def _get_f(self):
        """将矩阵统计成频率表 (float32, CPU)"""
        f = [
            [sum(each[i]) / len(each[i]) for i in range(self.num_expert)]
            for each in self.matrix
        ]
        return torch.tensor(f, dtype=torch.float32)  # 默认CPU, float32

    def update_matrix(self, expert_responses):
        """
        expert_responses: 1D tensor (GPU/CPU)
        更新matrix计数
        """
        expert_responses = expert_responses.detach().cpu().numpy()  # 仅用于计数
        for i in range(self.num_expert):
            for j in range(self.num_expert):
                if len(self.matrix[i][j]) > self.k:
                    self.matrix[i][j].pop(0)
                if expert_responses[i] > self.positive_threshold and expert_responses[j] > self.positive_threshold:
                    self.matrix[i][j].append(1)
                else:
                    self.matrix[i][j].append(0)
        return self._get_f()  # 返回CPU tensor

    def inference_simulation(self, expert_response):
        """
        expert_response: 1D tensor (device = cuda/cpu)
        使用链式推理模拟输出
        """
        device = expert_response.device
        dtype = expert_response.dtype

        x = expert_response.clone()
        m = self._get_f().to(device=device, dtype=dtype)

        for _ in range(self.num_expert - 1):
            x = x @ m  # 链式逻辑推理

        return x  # 与 expert_response 同 device, dtype

    def calc_loss(self, expert_response):
        """
        expert_response: 1D tensor (device = cuda/cpu)
        计算逻辑一致性损失 (MSE)
        """
        derived_expert_response = self.inference_simulation(expert_response)
        loss = torch.mean((expert_response - derived_expert_response) ** 2)
        return loss

    def plot_relation_matrix(self):
        """
        从 self.matrix 计算频率矩阵并绘制热力图
        """
        # 计算频率矩阵 f[i,j] = 关系成立次数 / 总次数
        f = np.array([
            [sum(self.matrix[i][j]) / len(self.matrix[i][j]) for j in range(self.num_expert)]
            for i in range(self.num_expert)
        ])

        plt.figure(figsize=(6, 5))
        plt.imshow(f, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Relation Strength P(Ej=1|Ei=1)")
        plt.title("Expert Relation Matrix (from NARSSimulator.matrix)")
        plt.xlabel("Expert j")
        plt.ylabel("Expert i")
        plt.show()


if __name__ == "__main__":
    NS = NARSSimulator(3, 2)
    print(NS.calc_loss(torch.tensor([0.6, 0.8, 0.2])))
