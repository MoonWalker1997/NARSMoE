import torch
import torch.nn as nn


class ExpertCNN(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_embed = nn.Linear(16 * 7 * 7, embedding_dim)

        self.probe = nn.Sequential(
            nn.Linear(embedding_dim, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
        for p in self.probe.parameters():
            p.requires_grad = False

    def forward(self, x):
        h = self.feature(x)
        h = h.view(h.size(0), -1)
        embed = self.fc_embed(h)

        activation = self.probe(embed)

        return embed, activation


class ExpertSystem(nn.Module):

    def __init__(self, num_experts=10, embedding_dim=32, num_classes=10):
        super().__init__()
        self.experts = nn.ModuleList([ExpertCNN(embedding_dim) for _ in range(num_experts)])
        self.classifier = nn.Sequential(
            nn.Linear(num_experts, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        expert_embeds = []
        expert_activations = []

        for expert in self.experts:
            embed, probe_out = expert(x)  # embed: [B, D], probe_out: [B, 1]
            expert_embeds.append(embed)
            expert_activations.append(probe_out.squeeze(1))

        expert_activations = torch.stack(expert_activations, dim=1)
        out = self.classifier(expert_activations)
        return out, expert_activations, expert_embeds
