"""
此脚本负责 - 定义Replay以及如何训练DQN
"""
import os
import datetime
from collections import namedtuple
import random

import numpy as np
import torch
import torch.nn as nn

from game.tetris import block
from model.cnn_model import DQN

# 俄罗斯方块矩阵的高度和宽度
row_count, col_count = 20, 10

MAX_Batch_Size = 51200
Replay_Capacity = 51200 * 3


current_path = os.path.dirname(os.path.abspath(__file__))
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.added_item_count = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        self.added_item_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = ReplayMemory(Replay_Capacity)


def train_DQN():
    gamma = 0.95

    # 0 -> 左移动
    # 1 -> 右移动
    # 2 -> 旋转
    # 3 -> 往下移动
    # 4 -> 往下移动到底部
    # action_space = [0, 1, 2, 3, 4]

    # 直接掉落到底部，局面很快就终结
    action_space = [0, 1, 2, 3]

    episodes = 10000

    model = DQN(row_count, col_count, len(action_space))
    loss_fn = nn.SmoothL1Loss()
    opt = torch.optim.RMSprop(model.parameters())

    print("############## Start Training", datetime.datetime.now())

    # 运行10局游戏
    for episode in range(episodes):
        board_state = np.zeros((row_count, col_count), np.ubyte)
        game_block = block()
        game_state, bloc_state = game_block.preview_init(board_state)

        # 每局游戏最多1000步
        for _ in range(1000):
            explore_exploit_tradeoff = np.random.uniform()

            action = np.random.choice(action_space)
            (
                stopflag,
                reward,
                created_block,
                newstate,
                dbginfo,
                board_state,
            ) = game_block.step(action, board_state)

            if stopflag:
                # 此局游戏结束
                break

            if newstate is not None:
                # 存入mem cache中去
                memory.push(game_state, action, newstate, reward)

                # tensor = torch.from_numpy(game_state)
                # tensor = tensor.unsqueeze(0)
                # tensor = tensor.unsqueeze(0)
                # tensor = tensor.float()
                # pred_q = model(tensor)[0][action]

                # new_tensor = torch.from_numpy(newstate)
                # new_tensor = new_tensor.unsqueeze(0)
                # new_tensor = new_tensor.unsqueeze(0)
                # new_tensor = tensor.float()
                # new_q = model(new_tensor).max(1).values[0].item()
                # # detach后的tensor和用torch.tensor(xxx)生成的tensor一样
                # # print(model(new_tensor).max(1)[0].detach())

                # if reward > 10:
                #     target_q = reward * 1.0
                # else:
                #     # target_q = pred_q.item() + lr * (
                #     #     reward + gamma * new_q - pred_q.item()
                #     # )
                #     target_q = reward + gamma * new_q

                # l = loss_fn(pred_q, torch.tensor(target_q))
                # opt.zero_grad()
                # l.backward()
                # # pytorch 的实例代码里有这么一段
                # for param in model.parameters():
                #     param.grad.data.clamp_(-1, 1)
                # opt.step()

                game_state = newstate
            else:
                # 存入mem cache中去
                memory.push(game_state, action, game_state, reward)

                # 往左移动、往右移动、旋转没反应的时候，设定其目标Q值为0
                # tensor = torch.from_numpy(game_state)
                # tensor = tensor.unsqueeze(0)
                # tensor = tensor.unsqueeze(0)
                # tensor = tensor.float()
                # pred_q = model(tensor)[0][action]
                # l = loss_fn(pred_q, torch.tensor(0.0))
                # opt.zero_grad()
                # l.backward()
                # # pytorch 的实例代码里有这么一段
                # for param in model.parameters():
                #     param.grad.data.clamp_(-1, 1)
                # opt.step()

            if (
                memory.added_item_count != 0
                and memory.added_item_count % MAX_Batch_Size == 0
            ):
                BATCH_SIZE = MAX_Batch_Size
                if len(memory) < BATCH_SIZE:
                    BATCH_SIZE = len(memory)
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                print(f"added_item_count is {memory.added_item_count}")
                memory.added_item_count = 0

                state_batch_list = []
                for tmp_state in batch.state:
                    ts = torch.from_numpy(tmp_state)
                    ts = ts.unsqueeze(0)
                    ts = ts.unsqueeze(0)
                    ts = ts.float()
                    state_batch_list.append(ts)
                state_batch = torch.cat(state_batch_list)

                action_batch = torch.tensor(
                    [[act] for act in batch.action], dtype=torch.int64
                )
                reward_batch = torch.tensor([[rwd] for rwd in batch.reward])
                print(reward_batch, reward_batch[reward_batch > 0])

                # print(action_batch)
                # print(model(state_batch))

                state_action_values = model(state_batch).gather(1, action_batch)

                next_state_batch_list = []
                for tmp_state in batch.next_state:
                    ts = torch.from_numpy(tmp_state)
                    ts = ts.unsqueeze(0)
                    ts = ts.unsqueeze(0)
                    ts = ts.float()
                    next_state_batch_list.append(ts)
                non_final_next_states = torch.cat(next_state_batch_list)
                next_state_values = torch.zeros(BATCH_SIZE).float()
                next_state_values = (
                    model(non_final_next_states).max(1)[0].unsqueeze(1).detach()
                )
                expected_state_action_values = next_state_values * gamma + reward_batch

                l = loss_fn(state_action_values, expected_state_action_values)
                opt.zero_grad()
                l.backward()
                # pytorch 的实例代码里有这么一段
                for param in model.parameters():
                    param.grad.data.clamp_(-1, 1)
                opt.step()

            if created_block is not None:
                game_block = created_block

    filename = f"Tetris_{episodes}.pt"
    torch.save(model.state_dict(), os.path.join("./outputs/", filename))
    print("############## End Training", datetime.datetime.now())


if __name__ == "__main__":
    train_DQN()
