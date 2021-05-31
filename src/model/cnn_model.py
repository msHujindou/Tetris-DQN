"""
此脚本负责 - 定义cnn model
"""
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, height: int, width: int, action_space_length: int):
        super(DQN, self).__init__()
        self.kernel_size = 3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=self.kernel_size, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.kernel_size, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=self.kernel_size, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = convw * convh * 32
        print("input size of full-connect layer is", linear_input_size)

        self.head = nn.Linear(linear_input_size, action_space_length)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 一层卷积
        x = F.relu(self.bn2(self.conv2(x)))  # 两层卷积
        x = F.relu(self.bn3(self.conv3(x)))  # 三层卷积
        return self.head(x.view(x.size(0), -1))  # 全连接层
