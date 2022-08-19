from modules.TweezerImageProcessing import get_xyza
import multiprocessing as mp

q = mp.Queue()

def f(x):
    # z = q.get()
    # print(z)
    for i in range(200):
        # ---bonus: gradually use up RAM---
        x += 10000  # linear growth; use exponential for faster ending: x *= 1.01
        y = list(range(int(x)))
        # ---------------------------------
    return x

if __name__ == '__main__':  # name guard to avoid recursive fork on Windows

    n = mp.cpu_count()  # multiply guard against counting only active cores
    for i in range(n):
        q.put(i+5)

    print('starting pool')
    with mp.Pool(n) as p:
        result = p.map(f, range(n))
    print(result)