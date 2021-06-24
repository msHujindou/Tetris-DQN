"""
此脚本是为了训练 q function, 计算下一个可用的q-table生成的时间
假定局面是4x10，俄罗斯方块仅仅只有正方形
4x10，且仅有正方形，总共有10000多种状态，episodes设置成100000可以完美训练出来
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

    episodes = 100000

    conf_last_episode = 0.99
    decay_rate = -np.log((conf_last_episode - min_eps) / (max_eps - min_eps)) / episodes

    print(decay_rate)

    print("#### Start training,", datetime.datetime.now())

    env = tetris_engine([Block_Type.L, Block_Type.T])
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
                action = np.argmax(qtable[game_state_key])
                action_name = env.action_type_list[action]
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
            print(f"episode {episode}'s reward : {total_reward_each_episode}")
            last_max_reward = total_reward_each_episode

        epsilon = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * episode)

    print("#### End training,", datetime.datetime.now())
    print("#### Total States Count", len(qtable))
    with open(f"./outputs/q_{episodes}.json", "w") as outfile:
        json.dump(qtable, outfile)


if __name__ == "__main__":
    train_Q_function()
