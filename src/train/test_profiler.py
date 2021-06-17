"""
train_dqn4训练成功，现有的CNN模型可以识别活着的俄罗斯方块距离右边、下边的距离
此脚本将对testcnn_reward做出更改，判断CNN是否能够识别出活着的方块距离障碍物的距离
"""
import os
import datetime


episodes_total = 6000
episodes_each_process = 100


def sample_data(p_episodes):
    final = 0
    for i in range(p_episodes):
        for j in range(1000):
            for k in range(1000):
                if j > k:
                    final += 1
                else:
                    final -= 1
    return final


import multiprocessing as mp
from pstats import SortKey


def train_DQN():
    cpu_count = mp.cpu_count()

    print("############## Start Training", datetime.datetime.now())

    for _ in range(episodes_total // (cpu_count * episodes_each_process)):
        sample_data(episodes_each_process)

    # with mp.Pool(processes=cpu_count) as pool:
    #     for _ in range(episodes_total // (cpu_count * episodes_each_process)):
    #         task_list = [episodes_each_process for _ in range(cpu_count)]
    #         res = pool.map(sample_data, task_list)

    print("############## End Training", datetime.datetime.now())


if __name__ == "__main__":
    import cProfile, pstats, io

    pr = cProfile.Profile()
    pr.enable()

    train_DQN()

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
