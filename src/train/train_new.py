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

Run 82 test_1625030677_d44e4e87 的结果表明
7x10局面，且仅有田字形的俄罗斯方块，episode设置成8000000，double dqn结构，训练出来的model完全没法用

Run 89 test_1625129311_4d2cd398 的结果表明
7x10局面，且仅有田字形的俄罗斯方块，episode设置成8000000, 非 double dqn，训练出来的model完全没法用
但比double dqn训练出来的稍微好一点，移动俄罗斯方块时predicition value至少能变化

Run 106 test_1627873129_40374f63 的结果表明
20x10局面，且仅有田字形的俄罗斯方块，double dqn，训练出来的model会使俄罗斯方块左右无限震荡

Run 107 test_1628158554_526b377a 的结果表明
20x10局面，且仅有田字形的俄罗斯方块，double dqn，训练出来的model无法消除哪怕一行

Run 110,111 的结果：
7x10，仅有田字型俄罗斯方块，非double dqn训练出来的结果会使俄罗斯方块仅仅往下移动
double dqn会使俄罗斯方块左右移动，episode设置成4000000仍无法消除一行

Run 112 的结果：
7x10，仅有田字型俄罗斯方块，double dqn，episode设置成4000000
把policy_net的更新频率降低后，效果下降

Run 116 结果表明如下会轻微提升model效果
1, 增加了replay的容量 ; 2, 缩短了policy_net的更新周期

Run 121 的结果如下
将operation_not_allowed_reward以及会导致game_over的action的reward的惩罚值降低至50，model效果提升明显。

Run 121/122/123 的共同结果表明
double dqn, target_net的更新频率提高的话model效果急剧恶化，更新频率降低的话model效果会提升，至于提升的上限不知道，无限提高会不会造成恶化也不知道

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

MAX_Batch_Size = 25600
Replay_Capacity = MAX_Batch_Size * 8


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


def sample_data(p_episodes, eps, p_net):
    if random.random() < 0.1:
        print(f"@@@@ pid[{os.getpid()}] p_episodes/eps is {p_episodes}/{eps}")
    env = tetris_engine(
        [Block_Type.O],
        [
            Action_Type.Left_Down,
            Action_Type.Right_Down,
            # Action_Type.Rotate_Down,
            Action_Type.Down,
        ],
    )
    res = []
    for _ in range(p_episodes):
        game_state = env.reset()
        # 每局游戏最多1600多步
        for _ in range(2000):
            explore_exploit_tradeoff = np.random.uniform()
            if explore_exploit_tradeoff > eps:
                with torch.no_grad():
                    action_index = (
                        p_net(
                            torch.from_numpy(game_state)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .float()
                        )
                        .max(1)[1]
                        .view(1, 1)
                    )
                action = env.select_random_step_action(action_index)
                # print(
                #     f"@@@@ pid[{os.getpid()}] model prediction used {action_index}/{action}"
                # )
            else:
                action_index, action = env.select_random_step()
            new_state, reward, is_game_end, debug = env.step(action)
            if is_game_end:
                # add punishment to operations that lead to game end
                res.append(
                    (game_state, action_index, None, Confs.game_end_punishment.value)
                )
                break
            res.append((game_state, action_index, new_state, reward))
            game_state = new_state
    return res


def train_DQN():
    gamma = 0.95
    min_eps = 0.001
    max_eps = 1.0
    conf_last_episode = 0.75
    epsilon = 1.0

    cpu_count = mp.cpu_count()

    policy_net = DQN(Confs.row_count.value + 1, Confs.col_count.value, 3)
    # 加入 double DQN 结构
    target_net = DQN(Confs.row_count.value + 1, Confs.col_count.value, 3)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    loss_fn = nn.SmoothL1Loss()
    opt = torch.optim.RMSprop(policy_net.parameters())

    total_iteration = episodes_total // (cpu_count * episodes_each_process)
    decay_rate = (
        -np.log((conf_last_episode - min_eps) / (max_eps - min_eps)) / total_iteration
    )

    print("#### mp.cpu_count() =", cpu_count)
    print("#### iteration count =", total_iteration)
    print("#### decay_rate =", decay_rate)
    print("#### Start Training", datetime.datetime.now())

    # 如果不设置spawn模式，在Linux环境下同一批次模拟出来的结果，完全一样，
    mp.set_start_method("spawn")

    with mp.Pool(processes=cpu_count) as pool:
        for _ in range(total_iteration):
            task_list = [
                (episodes_each_process, epsilon, policy_net) for _ in range(cpu_count)
            ]
            res = pool.starmap(sample_data, task_list)

            for itm_lst in res:
                # there's no need print debug information
                if _ <= 0:
                    print("---- state list size", len(itm_lst))
                #     print("#### size of state list is", len(itm_lst))
                #     tmpidx = -1
                #     for tmpitem in itm_lst:
                #         tmpidx += 1
                #         if tmpitem[3] > 0:
                #             print(itm_lst[tmpidx - 1][0])
                #             print("action :", itm_lst[tmpidx - 1][1])
                #             print(itm_lst[tmpidx - 1][2])
                #             print("reward :", itm_lst[tmpidx - 1][3])
                #             print("----")
                #             print(tmpitem[0])
                #             print("action :", tmpitem[1])
                #             print(tmpitem[2])
                #             print("reward :", tmpitem[3])
                #             break

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

                        non_final_mask = torch.tensor(
                            tuple(map(lambda s: s is not None, batch.next_state)),
                            dtype=torch.bool,
                        )
                        non_final_next_states = torch.cat(
                            [
                                torch.from_numpy(s).unsqueeze(0).unsqueeze(0).float()
                                for s in batch.next_state
                                if s is not None
                            ]
                        )

                        # print(non_final_mask)
                        # print(non_final_mask.size())
                        # print((non_final_mask == True).sum())
                        # print((non_final_mask == False).sum())

                        # below comment come from pytorch DQN example
                        # state_batch.shape torch.Size([128, 3, 40, 90])
                        # action_batch.shape torch.Size([128, 1])
                        # reward_batch.shape torch.Size([128])
                        # state_action_values.shape torch.Size([128, 1])
                        # expected_state_action_values.unsqueeze(1).shape torch.Size([128, 1])

                        state_batch = torch.cat(
                            [
                                torch.from_numpy(s).unsqueeze(0).unsqueeze(0).float()
                                for s in batch.state
                            ]
                        )

                        action_batch = torch.cat(
                            [
                                torch.tensor([[act]], dtype=torch.long)
                                for act in batch.action
                            ]
                        )

                        reward_batch = torch.cat(
                            [
                                torch.tensor([rwd], dtype=torch.float)
                                for rwd in batch.reward
                            ]
                        )

                        state_action_values = policy_net(state_batch).gather(
                            1, action_batch
                        )

                        next_state_values = torch.zeros(BATCH_SIZE)
                        next_state_values[non_final_mask] = (
                            target_net(non_final_next_states).max(1)[0].detach()
                        )
                        expected_state_action_values = (
                            next_state_values * gamma + reward_batch
                        )

                        # print(torch.min(expected_state_action_values))
                        # m = torch.min(expected_state_action_values)
                        # print((expected_state_action_values == m).sum())

                        # print("state_batch.shape", state_batch.shape)
                        # print("action_batch.shape", action_batch.shape)
                        # print("reward_batch.shape", reward_batch.shape)
                        # print("state_action_values.shape", state_action_values.shape)
                        # print(
                        #     "expected_state_action_values.unsqueeze(1).shape",
                        #     expected_state_action_values.unsqueeze(1).shape,
                        # )

                        if random.random() < 0.001:
                            print("#### Current Datetime:", datetime.datetime.now())
                            print(state_action_values)
                            print(expected_state_action_values)
                            # print(
                            #     "expect : max value ",
                            #     torch.max(expected_state_action_values),
                            # )
                            # print(
                            #     "expect : avg value ",
                            #     torch.mean(expected_state_action_values),
                            # )
                            # print(
                            #     "predict : max value ",
                            #     torch.max(state_action_values),
                            # )
                            # print(
                            #     "predict : avg value ",
                            #     torch.mean(state_action_values),
                            # )

                        l = loss_fn(
                            state_action_values,
                            expected_state_action_values.unsqueeze(1),
                        )
                        opt.zero_grad()
                        l.backward()

                        # pytorch 的实例代码里有这么一段
                        for param in policy_net.parameters():
                            param.grad.data.clamp_(-1, 1)
                        opt.step()

            if _ > 0 and _ % 15 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            epsilon = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * _)

    filename = f"Tetris_{episodes_total}.pt"
    torch.save(policy_net.state_dict(), os.path.join("./outputs/", filename))
    print("############## End Training", datetime.datetime.now())


if __name__ == "__main__":
    train_DQN()
