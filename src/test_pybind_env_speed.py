import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from .env import HanabiEnv


class Thread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return


def run_env_pybind(env):
    env.reset()
    obs = env.observe(0)

    while not env.terminated():
        obs = [env.observe(i) for i in range(2)]
        legal_move = obs[env.hle_state.cur_player()]["legal_move"]
        legal_move[5:10] = 0
        action = legal_move.float().multinomial(1).item()
        move = env.hle_game.get_move(action)
        env.step(move)
    return env.num_step


def run_env_loop_pybind(num_game):
    env = HanabiEnv(num_player=2, seed=1, max_len=-1, bomb=1)
    num_step = 0

    for i in range(num_game):
        num_step += run_env_pybind(env)

    return num_step


def run_env_loop_pybind_multithread(num_game_per_thread, num_thread):
    futs = []
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        for _ in range(num_thread):
            futs.append(executor.submit(lambda: run_env_loop_pybind(num_game_per_thread)))

    steps = [f.result() for f in futs]
    return np.sum(steps)


if __name__ == "__main__":
    import sys
    print(f"nogil={getattr(sys.flags, 'nogil', False)}")

    for num_t in [1, 2, 4, 8, 16, 32]:
        t = time.time()
        num_step = run_env_loop_pybind_multithread(100, num_t)
        end_t = time.time()
        print(f"{num_t} thread {num_step / (end_t - t):.2f} steps/s")
