"""
此脚本是为了训练 q function, 计算下一个可用的q-table生成的时间
假定局面是4x10，俄罗斯方块仅仅只有正方形
4x10，且仅有正方形，总共有10000多种状态，episodes设置成100000可以完美训练出来
5x10，且仅有正方形，总共有20000多种状态，全部设置成explorer训练，可以得到完美的model

5x10，且仅有山、L形，总共有360万种状态，episodes设置成1000000训练出来的模型完全没法用
请参考 Run70 test_1624527825_d36f748b

5x10, 且仅有山形，episodes设置成5000000，得到的state状态个数为4969649，此model可以消除两行
请参考 Run71 test_1624534454_2975f794

Run 80 test_1624964465_87c7a779 的训练结果如下：
局面为 5x10, 且仅有山形俄罗斯方块，episodes设置成18000000，得到的state状态为9908083，
此model最高可以消除5行，若从头开始的话，最多可以消除4行。

"""
import datetime
import numpy as np
import json
from game.confs import Block_Type
from game.tetris_engine import tetris_engine


def train_Q_function():
    lr = 0.8
    min_eps = 0.001
    max_eps = 1.0
    gamma = 0.95

    epsilon = 1.0
    qtable = {}

    episodes = 36000000

    conf_last_episode = 0.5
    decay_rate = -np.log((conf_last_episode - min_eps) / (max_eps - min_eps)) / episodes

    print(decay_rate)

    print("#### Start training,", datetime.datetime.now())

    env = tetris_engine([Block_Type.L])
    last_max_reward = 0

    # 运行10局游戏
    for episode in range(episodes):
        game_state = env.reset()
        game_state_key = game_state.tobytes().hex()

        total_reward_each_episode = 0

        for _ in range(2000):
            explore_exploit_tradeoff = np.random.uniform()
            if explore_exploit_tradeoff > epsilon:
                if game_state_key not in qtable:
                    qtable[game_state_key] = [0] * env.action_space
                # 某些状态下某些操作是被禁止的，比如靠近边界后的旋转，这个时候需要随机选择一个action
                if np.max(qtable[game_state_key]) > 0:
                    action = np.argmax(qtable[game_state_key])
                    action_name = env.action_type_list[action]
                else:
                    action, action_name = env.select_random_step()
            else:
                action, action_name = env.select_random_step()

            new_state, reward, done, debug = env.step(action_name)
            new_state_key = new_state.tobytes().hex()

            if done:
                # 此局游戏结束
                break

            total_reward_each_episode += reward

            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if game_state_key not in qtable:
                qtable[game_state_key] = [0] * env.action_space
            if new_state_key not in qtable:
                qtable[new_state_key] = [0] * env.action_space
            qtable[game_state_key][action] = qtable[game_state_key][action] + lr * (
                reward
                + gamma * np.amax(qtable[new_state_key])
                - qtable[game_state_key][action]
            )

            game_state = new_state
            game_state_key = game_state.tobytes().hex()

        if total_reward_each_episode > last_max_reward:
            print(
                f"@@@@ {datetime.datetime.now()} episode {episode}'s reward : {total_reward_each_episode}"
            )
            last_max_reward = total_reward_each_episode

        epsilon = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * episode)

    print("#### End training,", datetime.datetime.now())
    print("#### Total States Count", len(qtable))
    with open(f"./outputs/q_{episodes}.json", "w") as outfile:
        json.dump(qtable, outfile)


if __name__ == "__main__":
    train_Q_function()
