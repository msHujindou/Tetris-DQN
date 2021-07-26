"""
基于train_dqn9，添加如下更改：
1. 训练由state转换成灰色图片
2. 局面由 20x10 变为 21x10
3. 因为内存有限，batch设置成2048就能导致16G机器崩溃
4. 因为内存有限，每个进程的局数也没法设置高

将episode设置成200000，训练了160+小时，仍然没有出结果，改成20000
将MAX_Batch_Size设置成128，会出现内存不够错误

Run 102 test_1626833346_3eac8b9c 结果如下：
episode设置成10000需要的训练时间为75小时，训练出来的模型对识别左右有较好的效果，
对识别旋转效果一般般，对识别到底部的距离不理想

"""
import os
import datetime
from collections import namedtuple
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from game.confs import Action_Type, Confs
from game.tetris_engine import tetris_engine

from model.cnn_model import DQN
import multiprocessing as mp

from utils.util import create_image_from_state

MAX_Batch_Size = 64
Replay_Capacity = 64


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

episodes_total = 10000
episodes_each_process = 5


def convert_state_to_image(p_state):
    img = create_image_from_state(p_state)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sample_data(p_episodes):
    env = tetris_engine()
    res = []
    for _ in range(p_episodes):
        game_state = env.reset()
        # 每局游戏最多1600多步
        for _ in range(2000):
            action_index, action = env.select_random_step()
            r1 = env.test_step(Action_Type.Left)
            r2 = env.test_step(Action_Type.Right)
            r3 = env.test_step(Action_Type.Rotate)
            r4 = env.test_step(Action_Type.Down)
            new_state, reward, done, debug = env.step(action)
            if done:
                break
            res.append(
                (
                    convert_state_to_image(game_state),
                    action_index,
                    convert_state_to_image(new_state),
                    (r1, r2, r3, r4),
                )
            )
            game_state = new_state
    return res


def train_DQN():
    cpu_count = mp.cpu_count()

    model = DQN(292, 138, 4)
    loss_fn = nn.SmoothL1Loss()
    opt = torch.optim.RMSprop(model.parameters())

    print("############## mp.cpu_count() =", cpu_count)
    print(
        "############## iteration count =",
        episodes_total // (cpu_count * episodes_each_process),
    )
    print("############## Start Training", datetime.datetime.now())

    # 如果不设置spawn模式，在Linux环境下同一批次模拟出来的结果，完全一样，
    mp.set_start_method("spawn")

    with mp.Pool(processes=cpu_count) as pool:
        for _ in range(episodes_total // (cpu_count * episodes_each_process)):
            task_list = [episodes_each_process for _ in range(cpu_count)]
            res = pool.map(sample_data, task_list)

            for itm_lst in res:
                for itm in itm_lst:
                    memory.push(itm[0], itm[1], itm[2], itm[3])
                    if (
                        memory.added_item_count != 0
                        and memory.added_item_count % MAX_Batch_Size == 0
                    ):
                        BATCH_SIZE = MAX_Batch_Size
                        if len(memory) < BATCH_SIZE:
                            BATCH_SIZE = len(memory)
                        transitions = memory.sample(BATCH_SIZE)
                        batch = Transition(*zip(*transitions))
                        memory.added_item_count = 0

                        state_batch_list = []
                        for tmp_state in batch.state:
                            ts = torch.from_numpy(tmp_state)
                            ts = ts.unsqueeze(0)
                            ts = ts.unsqueeze(0)
                            ts = ts.float()
                            state_batch_list.append(ts)
                        state_batch = torch.cat(state_batch_list)

                        reward_batch = torch.tensor(
                            [[rwd[0], rwd[1], rwd[2], rwd[3]] for rwd in batch.reward]
                        ).float()

                        state_action_values = model(state_batch)

                        if random.random() < 0.01:
                            print("#### Current Datetime:", datetime.datetime.now())
                            print(reward_batch)
                            print(state_action_values)

                        expected_state_action_values = reward_batch

                        l = loss_fn(state_action_values, expected_state_action_values)
                        opt.zero_grad()
                        l.backward()

                        # pytorch 的实例代码里有这么一段
                        for param in model.parameters():
                            param.grad.data.clamp_(-1, 1)
                        opt.step()

    filename = f"Tetris_{episodes_total}.pt"
    torch.save(model.state_dict(), os.path.join("./outputs/", filename))
    print("############## End Training", datetime.datetime.now())


if __name__ == "__main__":
    train_DQN()
