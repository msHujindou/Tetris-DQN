"""
此脚本是为了检验train_q_function2.py总结的原因

总结原因：
4x10 仅有L型俄罗斯方块，的的确确存在数种无限循环，但每次所有方块都消除完后，
总会回归到那个初始状态，但消除行数的reward机制会促使避开无限循环

"""
import datetime
import numpy as np
from game.confs import Block_Type
from game.tetris_engine import tetris_engine

import sys
import cv2
from utils.util import create_image_from_state
from game.confs import Action_Type, Block_Type, Confs
from game.tetris_engine import tetris_engine
import json
import numpy as np

action_list = []


def sample():
    act_list = [
        1,
        1,
        3,
        3,
        3,
        3,
        1,
        1,
        1,
        2,
        2,
        3,
        3,
        3,
        3,
        0,
        0,
        3,
        3,
        3,
        3,
        2,
        2,
        0,
        3,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        2,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        2,
        2,
        2,
        3,
        3,
        3,
        1,
        1,
        3,
        3,
        3,
        3,
        1,
        1,
        1,
        2,
        2,
        3,
        3,
        3,
        3,
        0,
        0,
        3,
        3,
        3,
        3,
        2,
        2,
        0,
        3,
        3,
        3,
        3,
    ]
    act_list_not_finite = [
        3,
        3,
        3,
        3,
        1,
        1,
        1,
        3,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        3,
        3,
        3,
        3,
        0,
        0,
        0,
        2,
        2,
        3,
        3,
        3,
        3,
        2,
        1,
        3,
        3,
        3,
        1,
        1,
        1,
        2,
        1,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        3,
        3,
        3,
        3,
        0,
        3,
        3,
        3,
        3,
        1,
        1,
        1,
        2,
        3,
        3,
        3,
        0,
        0,
        0,
        2,
        3,
        3,
        3,
        2,
        3,
        3,
        3,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        3,
        2,
        3,
        3,
        0,
        2,
        3,
        3,
        3,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        1,
        1,
        1,
        3,
    ]
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
    game_state = env.reset()
    debug_img = None
    is_end = False
    tmplist = act_list.copy()
    itr_count = 0
    while True:
        img = create_image_from_state(game_state)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", img)
        if debug_img is not None:
            cv2.imshow("debug", debug_img)
        if len(tmplist) <= 0:
            itr_count += 1
            if itr_count % 2 == 0:
                tmplist = act_list.copy()
            else:
                tmplist = act_list_not_finite.copy()
        key = cv2.waitKey(10)

        # press Q or ESC
        if key == ord("q") or key == 27:
            break
        if key != ord("i"):
            tmpkey = tmplist.pop(0)
            if tmpkey == 0:
                key = ord("a")
            elif tmpkey == 1:
                key = ord("d")
            elif tmpkey == 2:
                key = ord("w")
            elif tmpkey == 3:
                key = ord("s")
            else:
                raise Exception("Key Error ...")

        if is_end:
            env = tetris_engine([Block_Type.L])
            game_state = env.reset()
            debug_img = None
            is_end = False
            if itr_count % 2 == 0:
                tmplist = act_list.copy()
            else:
                tmplist = act_list_not_finite.copy()
            continue

        game_state_key = game_state.tobytes().hex()
        if game_state_key not in qtable:
            qtable[game_state_key] = [0] * env.action_space

        if key == ord("w"):
            # rotate
            new_state, reward, is_end, debug = env.step(Action_Type.Rotate)
            new_state_key = new_state.tobytes().hex()
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if new_state_key not in qtable:
                qtable[new_state_key] = [0] * env.action_space
            # 对于无效的移动，不更新其Q value会使生成的Q table更健壮
            if new_state_key != game_state_key:
                qtable[game_state_key][2] = qtable[game_state_key][2] + lr * (
                    reward
                    + gamma * np.amax(qtable[new_state_key])
                    - qtable[game_state_key][2]
                )
            # print(np.array(list(qtable.values())))
            game_state = new_state
        elif key == ord("s"):
            # down
            new_state, reward, is_end, debug = env.step(Action_Type.Down)
            new_state_key = new_state.tobytes().hex()
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if is_end:
                qtable[game_state_key][3] = -100
            else:
                if new_state_key not in qtable:
                    qtable[new_state_key] = [0] * env.action_space
                # 对于无效的移动，不更新其Q value会使生成的Q table更健壮
                if new_state_key != game_state_key:
                    qtable[game_state_key][3] = qtable[game_state_key][3] + lr * (
                        reward
                        + gamma * np.amax(qtable[new_state_key])
                        - qtable[game_state_key][3]
                    )
            # print(np.array(list(qtable.values())))
            game_state = new_state
        elif key == ord("a"):
            # left
            new_state, reward, is_end, debug = env.step(Action_Type.Left)
            new_state_key = new_state.tobytes().hex()
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if new_state_key not in qtable:
                qtable[new_state_key] = [0] * env.action_space
            # 对于无效的移动，不更新其Q value会使生成的Q table更健壮
            if new_state_key != game_state_key:
                qtable[game_state_key][0] = qtable[game_state_key][0] + lr * (
                    reward
                    + gamma * np.amax(qtable[new_state_key])
                    - qtable[game_state_key][0]
                )
            # print(np.array(list(qtable.values())))
            game_state = new_state
        elif key == ord("d"):
            # right
            new_state, reward, is_end, debug = env.step(Action_Type.Right)
            new_state_key = new_state.tobytes().hex()
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if new_state_key not in qtable:
                qtable[new_state_key] = [0] * env.action_space
            # 对于无效的移动，不更新其Q value会使生成的Q table更健壮
            if new_state_key != game_state_key:
                qtable[game_state_key][1] = qtable[game_state_key][1] + lr * (
                    reward
                    + gamma * np.amax(qtable[new_state_key])
                    - qtable[game_state_key][1]
                )
            # print(np.array(list(qtable.values())))
            game_state = new_state
        elif key == ord(" "):
            # bottom
            game_state, reward, is_end, debug = env.step(Action_Type.Bottom)
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif key == ord("i"):
            print("#### iteration count is", itr_count)
            print(np.array(list(qtable.values())))

    cv2.destroyAllWindows()


def visual_train():
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
    game_state = env.reset()
    debug_img = None
    is_end = False
    while True:
        img = create_image_from_state(game_state)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", img)
        if debug_img is not None:
            cv2.imshow("debug", debug_img)
        key = cv2.waitKey(10)
        # press Q or ESC
        if key == ord("q") or key == 27:
            break

        if is_end:
            env = tetris_engine([Block_Type.L])
            game_state = env.reset()
            debug_img = None
            is_end = False
            continue

        game_state_key = game_state.tobytes().hex()
        if game_state_key not in qtable:
            qtable[game_state_key] = [0] * env.action_space

        if key == ord("w"):
            # rotate
            new_state, reward, is_end, debug = env.step(Action_Type.Rotate)
            new_state_key = new_state.tobytes().hex()
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if new_state_key not in qtable:
                qtable[new_state_key] = [0] * env.action_space
            # 对于无效的移动，不更新其Q value会使生成的Q table更健壮
            if new_state_key != game_state_key:
                qtable[game_state_key][2] = qtable[game_state_key][2] + lr * (
                    reward
                    + gamma * np.amax(qtable[new_state_key])
                    - qtable[game_state_key][2]
                )
            # print(np.array(list(qtable.values())))
            action_list.append(2)
            game_state = new_state
            print(game_state)
        elif key == ord("s"):
            # down
            new_state, reward, is_end, debug = env.step(Action_Type.Down)
            new_state_key = new_state.tobytes().hex()
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if new_state_key not in qtable:
                qtable[new_state_key] = [0] * env.action_space
            # 对于无效的移动，不更新其Q value会使生成的Q table更健壮
            if new_state_key != game_state_key:
                qtable[game_state_key][3] = qtable[game_state_key][3] + lr * (
                    reward
                    + gamma * np.amax(qtable[new_state_key])
                    - qtable[game_state_key][3]
                )
            # print(np.array(list(qtable.values())))
            action_list.append(3)
            game_state = new_state
            print(game_state)
        elif key == ord("a"):
            # left
            new_state, reward, is_end, debug = env.step(Action_Type.Left)
            new_state_key = new_state.tobytes().hex()
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if new_state_key not in qtable:
                qtable[new_state_key] = [0] * env.action_space
            # 对于无效的移动，不更新其Q value会使生成的Q table更健壮
            if new_state_key != game_state_key:
                qtable[game_state_key][0] = qtable[game_state_key][0] + lr * (
                    reward
                    + gamma * np.amax(qtable[new_state_key])
                    - qtable[game_state_key][0]
                )
            # print(np.array(list(qtable.values())))
            action_list.append(0)
            game_state = new_state
        elif key == ord("d"):
            # right
            new_state, reward, is_end, debug = env.step(Action_Type.Right)
            new_state_key = new_state.tobytes().hex()
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if new_state_key not in qtable:
                qtable[new_state_key] = [0] * env.action_space
            # 对于无效的移动，不更新其Q value会使生成的Q table更健壮
            if new_state_key != game_state_key:
                qtable[game_state_key][1] = qtable[game_state_key][1] + lr * (
                    reward
                    + gamma * np.amax(qtable[new_state_key])
                    - qtable[game_state_key][1]
                )
            # print(np.array(list(qtable.values())))
            action_list.append(1)
            game_state = new_state
        elif key == ord(" "):
            # bottom
            game_state, reward, is_end, debug = env.step(Action_Type.Bottom)
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            print(action_list)
            action_list.clear()
    cv2.destroyAllWindows()


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

            if done:
                # 此局游戏结束
                break

            new_state_key = new_state.tobytes().hex()

            total_reward_each_episode += reward

            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            if game_state_key not in qtable:
                qtable[game_state_key] = [0] * env.action_space
            if new_state_key not in qtable:
                qtable[new_state_key] = [0] * env.action_space

            # 对于无效的移动，不更新其Q value会使生成的Q table更健壮
            if new_state_key != game_state_key:
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


if __name__ == "__main__":
    # visual_train()
    sample()
