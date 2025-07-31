import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from NN_modules import ExpertSystem
from NARSSimulator import NARSSimulator  # 假设你把 NARSSimulator 放在 narss_module.py

# ====== 1. 数据加载 ======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# ====== 2. 初始化模型 & NARSSimulator ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExpertSystem(num_experts=10, embedding_dim=32).to(device)
narss = NARSSimulator(num_expert=10, k=5)  # NARS逻辑网络

# ====== 3. 定义优化器 & 损失函数 ======
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_cls = nn.CrossEntropyLoss()
lambda_logic = 0.1  # 逻辑一致性loss权重

# ====== 4. 训练循环 ======
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # ---- 前向传播 ----
        out, expert_acts, _ = model(x)  # expert_acts: [B, 10]

        # ---- 分类损失 ----
        loss_cls = criterion_cls(out, y)

        # ---- 逻辑一致性损失 (逐样本累积) ----
        loss_logic_batch = 0
        for i in range(expert_acts.size(0)):
            expert_response = expert_acts[i]  # ✅ shape [10]，单样本的专家激活
            loss_logic_batch += narss.calc_loss(expert_response)
            narss.update_matrix(expert_response)
        loss_logic_batch /= expert_acts.size(0)

        # ---- 总损失 ----
        loss = loss_cls + lambda_logic * loss_logic_batch

        # ---- 反向传播 & 参数更新 ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- 统计 ----
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(out, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    acc = total_correct / total_samples
    print(f"[Epoch {epoch+1}] Loss={total_loss/total_samples:.4f} Acc={acc:.4f}")

# ====== 5. 测试阶段 ======
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out, _, _ = model(x)
        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Test Accuracy: {correct/total:.4f}")
narss.plot_relation_matrix()
