import multiprocessing as mp
import time
import datetime


def f(x):
    time.sleep(5)
    return [x, x * x]


if __name__ == "__main__":
    cpu_count = mp.cpu_count()

    print("multiprocessing count is ", cpu_count, type(cpu_count))

    start_time = datetime.datetime.now()

    with mp.Pool(processes=cpu_count) as pool:
        res = pool.map(f, range(cpu_count))
        print(res)
        print(type(res))

    end_time = datetime.datetime.now()
    print(end_time - start_time)

    print("Now the pool is closed and no longer available")
