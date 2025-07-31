import random

import numpy as np
import torch
from matplotlib import pyplot as plt


class NARSSimulator:
    def __init__(self, num_expert, k):
        self.num_expert = num_expert
        self.k = k

        self.matrix = [
            [[random.choice([0, 1])] for _ in range(num_expert)]
            for _ in range(num_expert)
        ]

        self.positive_threshold = 0.5
        self.negative_threshold = 0.5

    def _get_f(self):
        f = [
            [sum(each[i]) / len(each[i]) for i in range(self.num_expert)]
            for each in self.matrix
        ]
        return torch.tensor(f, dtype=torch.float32)

    def update_matrix(self, expert_responses):
        expert_responses = expert_responses.detach().cpu().numpy()
        for i in range(self.num_expert):
            for j in range(self.num_expert):
                if len(self.matrix[i][j]) > self.k:
                    self.matrix[i][j].pop(0)
                if expert_responses[i] > self.positive_threshold and expert_responses[j] > self.positive_threshold:
                    self.matrix[i][j].append(1)
                else:
                    self.matrix[i][j].append(0)
        return self._get_f()

    def inference_simulation(self, expert_response):
        device = expert_response.device
        dtype = expert_response.dtype

        x = expert_response.clone()
        m = self._get_f().to(device=device, dtype=dtype)

        for _ in range(self.num_expert - 1):
            x = x @ m

        return x

    def calc_loss(self, expert_response):
        derived_expert_response = self.inference_simulation(expert_response)
        loss = torch.mean((expert_response - derived_expert_response) ** 2)
        return loss

    def plot_relation_matrix(self):
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
