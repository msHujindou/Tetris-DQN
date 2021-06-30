"""
输入的为20 * 10的矩阵，因此摒弃卷积、直接使用FC
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class DQN_FC(nn.Module):
    def __init__(self, height: int, width: int, action_space_length: int):
        super(DQN_FC, self).__init__()
        self.conv1 = nn.Linear(height * width, 256)
        self.conv2 = nn.Linear(256, 64)
        self.conv3 = nn.Linear(64, 16)
        self.head = nn.Linear(16, action_space_length)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 一层
        x = F.relu(self.conv2(x))  # 两层
        x = F.relu(self.conv3(x))  # 三层
        return self.head(x.view(x.size(0), -1))  # 全连接层


class DQN_FC2(nn.Module):
    def __init__(self, height: int, width: int, action_space_length: int):
        super(DQN_FC2, self).__init__()
        self.conv1 = nn.Linear(height * width, 256)
        self.conv2 = nn.Linear(256, 64)
        self.conv3 = nn.Linear(64, 16)
        self.head = nn.Linear(16, action_space_length)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 一层
        x = F.relu(self.conv2(x))  # 两层
        x = F.relu(self.conv3(x))  # 三层
        return self.head(x)  # 全连接层


# data = np.random.randint(0, high=255, size=(20, 10), dtype=np.ubyte)
# data2 = np.random.randint(0, high=255, size=(20, 10), dtype=np.ubyte)
# state_batch_list = []

# model = DQN_FC(20, 10, 4)
# model2 = DQN_FC(20, 10, 4)
# ts = torch.from_numpy(data.flatten()).float().unsqueeze(0)
# ts2 = torch.from_numpy(data2.flatten()).float().unsqueeze(0)
# state_batch_list.append(ts)
# state_batch_list.append(ts2)
# state_batch = torch.cat(state_batch_list)
# print(state_batch)
# print(model(state_batch))
# print(model2(state_batch))
