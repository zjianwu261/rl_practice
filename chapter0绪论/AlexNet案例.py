import torch
import torch.nn as nn
from torchinfo import summary


# 定义AlexNet的网络结构
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        # 定义卷积层
        self.features = nn.Sequential(
            # 卷积+Relu+最大池化
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 卷积+Relu+最大池化
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 卷积+Relu
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 卷积+ReLU
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 卷积+ReLU+最大池化
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 定义全连接层
        self.classifier = nn.Sequential(
            # Dropout + 全连接层 + Relu
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # Dropout + 全连接层+ ReLU
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # 全连接层
            nn.Linear(4096, num_classes),
        )

    # 定义前向传播函数
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# summary(AlexNet(), input_size=(1, 3, 224, 224))

# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import *
import numpy as np
import sys

# 设备检测，若未检测到cuda设备则在CPU上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机种子
torch.manual_seed(0)
# 定义模型、优化器、损失函数
model = AlexNet(num_classes=102).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 设置训练集的数据交换，进行数据增强
trainfom_train = transforms.Compose(
    {
        transforms.RandomRotation(30),  # 随机旋转-30度到30度之间
        transforms.RandomResizedCrop((224, 224)),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ToTensor(),  # 将图片转换为tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 归一化
    }
)
# 设置测试集的数据变换，不进行数据增强，仅使用resize和归一化
transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # resize
        transforms.ToTensor(),  # 将数据转换为张量
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 归一化
    ]
)
# 加载训练数据，需要特别注意的是Flowers102数据集，test簇的数据量较多些，所以这里使用"test"作为训练集
train_dataset = datasets.Flowers102(
    root="./data", split="test", transform=trainfom_train, download=True
)
# 实例化训练数据加载器
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
# 加载测试数据，使用“train”作为测试集
test_dataset = datasets.Flowers102(
    root="./data", split="train", transform=transform_test, download=True
)
# 实例化测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

# 设置epoch数开始训练
num_epochs = 500  # 设置epoch数
loss_history = []  # 创建损失历史记录列表
acc_history = []  # 创建准确率历史记录列表

# tqdm用于显示进度条并评估任务时间开销
for epoch in tqdm(range(num_epochs), file=sys.stdout):
    # 记录损失和预测正确数
    total_loss = 0
    total_correct = 0
    # 批量训练
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 预测、损失函数、反向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 记录训练集loss
        total_loss += loss.item()

    # 测试模型，不计算梯度
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 将数据转移到指定计算资源设备上
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 预测
            outputs = model(inputs)
            # 记录测试集预测正确数
            total_correct += (outputs.argmax(1) == labels).sum().item()

    # 记录训练集损失和测试集准确率
    loss_history.append(np.log10(total_loss))
    acc_history.append(total_correct / len(test_dataset))
    # 打印中间值
    if epoch % 50 == 0:
        tqdm.write(
            "Epoch:{0} Loss:{1} Acc:{2}".format(
                epoch, loss_history[-1], acc_history[-1]
            )
        )

# 使用matplotlib 绘制损失和准确率曲线
import matplotlib.pyplot as plt

plt.plot(loss_history, label="Loss")
plt.plot(acc_history, label="accuracy")
plt.legend()
plt.show()

# 输出准确率
print("Accuracy:", acc_history[-1])
