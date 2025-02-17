import torch
from torch import nn
from torch.nn import functional as F


# 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.right = shortcut

    def forward(self, x):
        output = self.left(x)
        # 对应网络中的实线和虚线，是否需要调整特征图维度
        residual = x if self.right is None else self.right(x)
        output += residual
        return F.relu(output)


def _make_layer(in_ch, out_ch, block_num, stride=1):
    # 维度增加时执行网络中的虚线，对shortcut 使用1*1矩阵增大维度
    shortcut = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
        nn.BatchNorm2d(out_ch),
    )
    layers = []
    layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))
    # 只有第一个shortcut需要统一特征图维度。

    for i in range(1, block_num):
        layers.append(ResidualBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        # 第一层,输入为224*224*3。
        self.pre = nn.Sequential(
            # (224+p*2-7)/s+1=(224+6-7)/2+1=112
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 输入为112*112*64，输出为56*56*64
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        )

        # 四个layer分别有3，4，6，3个residual block
        self.layer1 = _make_layer(64, 64, 2)  # 56*56*64
        self.layer2 = _make_layer(128, 128, 2, stride=1)  # 28*28*128 stride=2在第一层实现下采样
        self.layer3 = _make_layer(256, 256, 2, stride=1)  # 14*14*256
        self.layer4 = _make_layer(512, 512, 2, stride=1)  # 7*7*512
        # 最终的全连接层
        self.Conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
        self.fc = nn.Linear(512, num_classes)  # 7*7*512使用全局平均池化

    def forward(self, x):
        out = []
        x = self.pre(x)
        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)
        # x2 = self.la
        # x = self.Conv(x)
        # x = F.avg_pool2d(x, 7)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return out
