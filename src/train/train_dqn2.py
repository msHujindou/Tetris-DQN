"""
修改reward，测试CNN是否能够识别活着的俄罗斯方块距离右边、底部的距离
"""
import os
import datetime
from collections import namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
from game.confs import Confs
from game.tetris_engine import tetris_engine

from model.cnn_model import DQN

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


def testcnn_reward(new_state):
    x = Confs.col_count.value
    y = Confs.row_count.value
    for tmpx in range(Confs.col_count.value):
        if np.any(new_state[:, Confs.col_count.value - 1 - tmpx] == 128):
            x = tmpx
            break
    for tmpy in range(Confs.row_count.value):
        if np.any(new_state[Confs.row_count.value - 1 - tmpy, :] == 128):
            y = tmpy
            break
    return x, y


def train_DQN():
    gamma = 0.95

    env = tetris_engine()

    episodes = 100000

    model = DQN(Confs.row_count.value, Confs.col_count.value, env.action_space)
    loss_fn = nn.SmoothL1Loss()
    opt = torch.optim.RMSprop(model.parameters())

    print("############## Start Training", datetime.datetime.now())

    # 运行10局游戏
    for _ in range(episodes):
        game_state = env.reset()

        # 每局游戏最多1000步
        for _ in range(1000):
            action_index, action = env.select_random_step()
            new_state, reward, done, debug = env.step(action)

            rx, ry = testcnn_reward(new_state)

            if action_index % 2 == 0:
                reward = rx
            else:
                reward = ry

            if done:
                break

            memory.push(game_state, action_index, new_state, reward)
            game_state = new_state

            if (
                memory.added_item_count != 0
                and memory.added_item_count % MAX_Batch_Size == 0
            ):
                BATCH_SIZE = MAX_Batch_Size
                if len(memory) < BATCH_SIZE:
                    BATCH_SIZE = len(memory)
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                # print(f"added_item_count is {memory.added_item_count}")
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
                # print(reward_batch, reward_batch[reward_batch > 0])

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

    filename = f"Tetris_{episodes}.pt"
    torch.save(model.state_dict(), os.path.join("./outputs/", filename))
    print("############## End Training", datetime.datetime.now())


if __name__ == "__main__":
    train_DQN()
