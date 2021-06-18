"""
train_dqn6基础上，加上profiler并适当减少了episode数量
"""
import os
import datetime
from collections import namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
from game.confs import Action_Type, Confs
from game.tetris_engine import tetris_engine

from model.cnn_model import DQN
import multiprocessing as mp
from pstats import SortKey
import cProfile, pstats, io

MAX_Batch_Size = 51200 * 2
Replay_Capacity = 51200 * 2


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

episodes_total = 4000
episodes_each_process = 100


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
            res.append((game_state, action_index, new_state, (r1, r2, r3, r4)))
            game_state = new_state
    return res


def train_DQN():
    cpu_count = mp.cpu_count()

    model = DQN(Confs.row_count.value, Confs.col_count.value, 4)
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
                if _ <= 0:
                    print("#### size of state list is", len(itm_lst))
                    print(itm_lst[0])

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
                            print(state_batch[0, 0, :, :])
                            print(reward_batch[0, :])
                            print(state_action_values[0, :])

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
    pr = cProfile.Profile()
    pr.enable()

    train_DQN()

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
