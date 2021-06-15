"""
此脚本用来采样、统计在仅有左、右、下、旋转操作的情况下，每局需要多少步
"""
import os
import datetime
from game.tetris_engine import tetris_engine

current_path = os.path.dirname(os.path.abspath(__file__))


def train_DQN():
    """

    test_1623398527_b2193f61的运行结果：

    count    1000000.000000
    mean         891.897343
    std          167.805117
    min          233.000000
    25%          778.000000
    50%          895.000000
    75%         1007.000000
    max         1675.000000
    Name: StepCount, dtype: float64

        Raises:
            Exception: [description]
    """
    env = tetris_engine()

    episodes = 1000000

    print("############## Start Training", datetime.datetime.now())

    res_list = []

    res_list.append(
        ("StepCount", "Reward", "LeftCount", "RightCount", "RotateCount", "DownCount")
    )

    # 运行10局游戏
    for _ in range(episodes):
        game_state = env.reset()
        tmp_left = 0
        tmp_right = 0
        tmp_rotate = 0
        tmp_down = 0
        tmp_reward = 0

        # 每局游戏最多1000步
        for step in range(50000):
            action_index, action = env.select_random_step()
            new_state, reward, done, debug = env.step(action)
            tmp_reward += reward
            if action_index == 0:
                tmp_left += 1
            elif action_index == 1:
                tmp_right += 1
            elif action_index == 2:
                tmp_rotate += 1
            elif action_index == 3:
                tmp_down += 1
            else:
                raise Exception("Action Index > 3")

            if done:
                res_list.append(
                    (step, tmp_reward, tmp_left, tmp_right, tmp_rotate, tmp_down)
                )
                break

            game_state = new_state
        else:
            res_list.append(
                (step, tmp_reward, tmp_left, tmp_right, tmp_rotate, tmp_down)
            )

    with open("./outputs/statistics.csv", "w") as fw:
        for tmpitem in res_list:
            fw.write(f"{','.join(str(x) for x in tmpitem)}\n")

    print("############## End Training", datetime.datetime.now())


if __name__ == "__main__":
    train_DQN()
