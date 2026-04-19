# Homework 2 - CNN Classification

姓名：唐晓刚
学号：BC25038003

# 1. 导入所需要的库
### code
```python
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
```
## 设备选择，优先选择GPU运行
### code
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

# 2. 构建读取SVHN数据集的class
### code
```python
class SVHNDataset(Dataset):
    def __init__(self, mat_file):
        """
        读取 SVHN .mat 数据
        """
        data = sio.loadmat(mat_file)

        self.X = data['X']  # 原始图像数据 (32,32,3,N)
        self.y = data['y'].flatten() # 标签 (N, 1)

        # 标签处理：10 → 0（SVHN中标签10将其设置为0，保证在0~9之间）
        self.y[self.y == 10] = 0

        # 转换维度 → (N,3,32,32)，并归一化
        self.X = np.transpose(self.X, (3, 2, 0, 1)) / 255.0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """
        每次训练取一张图和标签
        """
        img = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img, label
```

# 2. 构建神经网络并定义损失函数和优化器
## 定义CNN神经网络
### code
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 卷积层1：3 → 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # 卷积层2：32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 池化层（降采样）
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        """
        前向传播
        """
        # 卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 → 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 → 8x8

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
```
## 定义损失函数和优化器
### code
```python
# 损失函数（分类任务标准）
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.005) #学习率设置为0.005，过小会过拟合
```


# 3. 数据加载并且初始化模型
### code
```python
train_dataset = SVHNDataset('train_32x32.mat')
test_dataset = SVHNDataset('test_32x32.mat')

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)


model = CNN().to(device)


train_losses = []
test_losses = []
train_accs = []
test_accs = []

```

# 4. 测试函数
### code
```python
def evaluate(loader):
    """
    在测试集上评估模型
    """
    model.eval()  # 进入评估模式

    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():  # 不计算梯度（节省计算）
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            loss_sum += loss.item()

            # 预测类别
            _, pred = torch.max(outputs, 1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / len(loader), correct / total

```

# 4. 开始训练
### code
```python
#训练循环20次
epochs = 20

for epoch in range(epochs):
    model.train()  # 进入训练模式

    running_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(x)

        # 计算损失
        loss = criterion(outputs, y)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计
        running_loss += loss.item()

        _, pred = torch.max(outputs, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    # 计算训练指标
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # 测试集评估
    test_loss, test_acc = evaluate(test_loader)

    # 记录
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # 输出
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

# 保存模型
torch.save(model.state_dict(), "svhn_cnn.pth")
```


# 5. 结果可视化
### code
```python
epochs_x = np.arange(1,epochs+1,1)
# Loss 曲线
plt.figure(figsize=(8,6))
plt.plot(epochs_x, train_losses, label='Train Loss',lw=2)
plt.plot(epochs_x, test_losses, label='Test Loss',lw=2)
plt.legend()
plt.title('Loss Curve', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0,epochs)
plt.savefig('Loss curve')
plt.show()

# Accuracy 曲线
plt.figure(figsize=(8,6))
plt.plot(epochs_x, train_accs, label='Train Accuracy',lw=2)
plt.plot(epochs_x, test_accs, label='Test Accuracy',lw=2)
plt.legend()
plt.title('Accuracy Curve', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0,epochs)
plt.savefig('Accuracy curve')
plt.show()
```
<img src="Loss curve.png" alt="alt text" width="800">
<img src="Accuracy curve.png" alt="alt text" width="800">
