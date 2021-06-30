"""
在train_dqn8以及train_dqn2的基础上，开发的此脚本

Run 67 test_1624278386_c5061593 的结果表明，
在左移动、右移动、旋转、向下的操作的情况下，训练的模型会导致俄罗斯方块反复在往左、往右切换

Run 69 test_1624448438_3dfb1ce2 的结果表明
20x10局面，且仅有山形的俄罗斯方块，episode设置成1200000，训练出来的model完全没法用

Run 75 test_1624537127_75a47871 的结果表明
10x10局面，且仅有田字形的俄罗斯方块，episode设置成800000，训练出来的model完全没法用，总是倾向往下移动

Run 78 test_1624619472_01b4cd06 的结果表明
7x10局面，且仅有田字形的俄罗斯方块，episode设置成1300000，训练出来的model基本接近没法用，
会倾向生成让其旋转的结果。

Run 81 test_1624968965_fdfa86a3 的结果表明
7x10局面，且仅有田字形的俄罗斯方块，episode设置成1300000，加入double dqn结构，训练出来的model接近没法用

"""
import os
import datetime
from collections import namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
from game.confs import Action_Type, Block_Type, Confs
from game.tetris_engine import tetris_engine

from model.cnn_model import DQN
import multiprocessing as mp

MAX_Batch_Size = 51200
Replay_Capacity = 51200 * 4


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

episodes_total = 8000000
episodes_each_process = 100


def sample_data(p_episodes):
    env = tetris_engine([Block_Type.O])
    res = []
    for _ in range(p_episodes):
        game_state = env.reset()
        # 每局游戏最多1600多步
        for _ in range(2000):
            action_index, action = env.select_random_step()
            new_state, reward, done, debug = env.step(action)
            if done:
                break
            res.append((game_state, action_index, new_state, reward))
            game_state = new_state
    return res


def train_DQN():
    gamma = 0.95
    cpu_count = mp.cpu_count()

    model = DQN(Confs.row_count.value, Confs.col_count.value, 4)
    # 加入 double DQN 结构
    target_net = DQN(Confs.row_count.value, Confs.col_count.value, 4)
    target_net.load_state_dict(model.state_dict())
    target_net.eval()

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
                    tmpidx = -1
                    for tmpitem in itm_lst:
                        tmpidx += 1
                        if tmpitem[3] > 0:
                            print(itm_lst[tmpidx - 1][0])
                            print("action :", itm_lst[tmpidx - 1][1])
                            print(itm_lst[tmpidx - 1][2])
                            print("reward :", itm_lst[tmpidx - 1][3])
                            print("----")
                            print(tmpitem[0])
                            print("action :", tmpitem[1])
                            print(tmpitem[2])
                            print("reward :", tmpitem[3])
                            break

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

                        action_batch = torch.tensor(
                            [[act] for act in batch.action], dtype=torch.int64
                        )

                        # mx = np.max(batch.reward)
                        # print("#### max reward", mx)
                        # print("#### average reward", np.average(batch.reward))
                        # print("#### count max", np.sum(batch.reward == mx))
                        # import sys
                        # sys.exit(0)

                        reward_batch = torch.tensor(
                            [[rwd] for rwd in batch.reward]
                        ).float()

                        state_action_values = model(state_batch).gather(1, action_batch)

                        next_state_batch_list = []
                        for tmp_state in batch.next_state:
                            ts = torch.from_numpy(tmp_state)
                            ts = ts.unsqueeze(0)
                            ts = ts.unsqueeze(0)
                            ts = ts.float()
                            next_state_batch_list.append(ts)
                        non_final_next_states = torch.cat(next_state_batch_list)
                        next_state_values = (
                            target_net(non_final_next_states)
                            .max(1)[0]
                            .detach()
                            .unsqueeze(1)
                        )
                        expected_state_action_values = (
                            next_state_values * gamma + reward_batch
                        )

                        if random.random() < 0.01:
                            print("#### Current Datetime:", datetime.datetime.now())
                            print(
                                "expect : max value ",
                                torch.max(expected_state_action_values),
                            )
                            print(
                                "expect : avg value ",
                                torch.mean(expected_state_action_values),
                            )
                            print(
                                "predict : max value ",
                                torch.max(state_action_values),
                            )
                            print(
                                "predict : avg value ",
                                torch.mean(state_action_values),
                            )

                        l = loss_fn(state_action_values, expected_state_action_values)
                        opt.zero_grad()
                        l.backward()

                        # pytorch 的实例代码里有这么一段
                        for param in model.parameters():
                            param.grad.data.clamp_(-1, 1)
                        opt.step()

            if _ > 0 and _ % 10 == 0:
                target_net.load_state_dict(model.state_dict())

    filename = f"Tetris_{episodes_total}.pt"
    torch.save(model.state_dict(), os.path.join("./outputs/", filename))
    print("############## End Training", datetime.datetime.now())


if __name__ == "__main__":
    train_DQN()
