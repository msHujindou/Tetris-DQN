"""
此脚本负责 - 训练基于Q_table的Q Function
"""
import datetime

import numpy as np

from game.tetris import block

# 俄罗斯方块矩阵的高度和宽度
row_count, col_count = 20, 10


def train_Q_function():
    lr = 0.8
    min_eps = 0.001
    max_eps = 1.0
    gamma = 0.95
    decay_rate = 0.000005

    rewards = []
    epsilon = 1.0
    qtable = {}

    # 0 -> 左移动
    # 1 -> 右移动
    # 2 -> 旋转
    # 3 -> 往下移动
    # 4 -> 往下移动到底部
    # action_space = [0, 1, 2, 3, 4]

    # 直接掉落到底部，局面很快就终结
    action_space = [0, 1, 2, 3]

    episodes = 1000000

    step_list_of_each_episode = []

    print(datetime.datetime.now())

    # 运行10局游戏
    for episode in range(episodes):
        start_time = datetime.datetime.now()

        board_state = np.zeros((row_count, col_count), np.ubyte)
        game_block = block()
        game_state, bloc_state = game_block.preview_init(board_state)

        # 生成运行图片
        # Path(f"./debug/{episode}").mkdir(parents=True, exist_ok=True)
        # debug_img = create_image_from_state(game_state)
        # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f"./debug/{episode}/{0}.png", debug_img)

        total_reward = 0

        # 每局游戏最多1000步
        for _ in range(1000):
            explore_exploit_tradeoff = np.random.uniform()
            if explore_exploit_tradeoff > epsilon:
                if game_state.tobytes() not in qtable:
                    qtable[game_state.tobytes()] = [0] * len(action_space)
                action = np.argmax(qtable[game_state.tobytes()])
            else:
                action = np.random.choice(action_space)

            (
                stopflag,
                reward,
                created_block,
                restate,
                dbginfo,
                board_state,
            ) = game_block.step(action, board_state)

            if stopflag:
                # 此局游戏结束
                step_list_of_each_episode.append(_)

                # 生成运行图片
                # debug_img = create_image_from_state(restate)
                # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                # cv2.imwrite(f"./debug/{episode}/{_ + 1}.png", debug_img)
                # debug_img = create_image_from_state(dbginfo)
                # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                # cv2.imwrite(f"./debug/{episode}/final.png", debug_img)

                break

            # if cleared_count > 0:
            #     print(f"episode {episode} cleared {cleared_count} lines")

            newstate = restate

            # if reward == 0:
            #     if action == 3:
            #         reward = 1
            #     elif action == 4:
            #         reward = 5
            # else:
            #     print(f"奇迹发生了, 消除了{cleared_count}行")

            if newstate is not None:
                # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                if game_state.tobytes() not in qtable:
                    qtable[game_state.tobytes()] = [0] * len(action_space)
                if newstate.tobytes() not in qtable:
                    qtable[newstate.tobytes()] = [0] * len(action_space)
                qtable[game_state.tobytes()][action] = qtable[game_state.tobytes()][
                    action
                ] + lr * (
                    reward
                    + gamma * np.amax(qtable[newstate.tobytes()])
                    - qtable[game_state.tobytes()][action]
                )
                game_state = newstate

                # 生成运行图片
                # debug_img = create_image_from_state(game_state)
                # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                # cv2.imwrite(f"./debug/{episode}/{_ + 1}.png", debug_img)

            total_reward += reward

            if created_block is not None:
                game_block = created_block

        epsilon = min_eps + (max_eps - min_eps) * np.exp(-decay_rate * episode)
        rewards.append(total_reward)

        # print(f"Used Time is {(datetime.datetime.now() - start_time).seconds}")
    print(f"Total Score: {np.sum(rewards)} , Reward List Length: {len(rewards)}")
    print("Average Score: ", np.sum(rewards) / episodes)
    print(f"State Count is {len(qtable)}")
    print(f"Min Step of each episode is {np.min(step_list_of_each_episode)}")
    print(f"Max Step of each episode is {np.max(step_list_of_each_episode)}")
    print(
        f"Average Step of each episode is {np.sum(step_list_of_each_episode)/ episodes}"
    )


if __name__ == "__main__":
    train_Q_function()
