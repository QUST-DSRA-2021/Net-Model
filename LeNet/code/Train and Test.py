import torch
from torch import nn, optim
from LeNet5 import LeNet
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

# 本程序使用 cpu 进行计算，GPU 版本尚未更新

# Data.py 文件中的 MyDataset 类存在一点小问题，但是可以正常使用。这里由于问题尚未解决使用 datasets.MNIST
# 第一次使用时 download 设置为 True
train_dataset = datasets.MNIST(root='../', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root='../', train=False, transform=transforms.ToTensor(), download=False)

# 训练集
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=False,
)

# 测试集
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
)

# 初始化模型
net = LeNet()

# 损失函数
loss_func = nn.CrossEntropyLoss()
# 梯度下降
optimizer = optim.SGD(net.parameters(), lr=0.1)

# 开始训练
# 迭代次数
EPOCH = 10
loss_l = []
for epoch in range(EPOCH):
    sum_loss = 0
    # 数据读取
    for i, data in enumerate(train_loader):
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        output = net.forward(inputs)
        loss = loss_func(output, labels)
        loss.backward()
        # 更新参数
        optimizer.step()

        # 每训练100个batch打印一次平均loss
        sum_loss += loss.item()
        if i % 100 == 99:
            print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))

    # 把每个epoch的 loss 存入列表，方便绘图
    loss_l.append(sum_loss)

    correct = 0
    total = 0
    for data in test_loader:
        test_inputs, labels = data
        outputs_test = net.forward(test_inputs)
        # 输出得分最高的类
        _, predicted = torch.max(outputs_test.data, 1)
        # 统计50个batch 图片的总个数
        total += labels.size(0)
        # 统计50个batch 正确分类的个数
        correct += (predicted == labels).sum()

    print('第{}个epoch的识别准确率为：{}%'.format(epoch + 1, 100 * correct.item() / total))


# 绘制 loss
plt.style.use("solarized-light")
plt.plot(range(EPOCH), loss_l)
#plt.show()
plt.savefig("./fig-loss.jpeg")
plt.clf()
