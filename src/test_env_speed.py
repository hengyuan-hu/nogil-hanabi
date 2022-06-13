import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from hanabi_learning_environment import rl_env


def run_env(env):
    obs = env.reset()
    num_step = 0
    while not env.state.is_terminal():
        cur_player = obs["current_player"]
        legal_move = obs["player_observations"][cur_player]["legal_moves"]
        legal_move = [m for m in legal_move if m["action_type"] != "PLAY"]
        move = np.random.choice(legal_move)
        obs = env.step(move)[0]
        num_step += 1
    return num_step


def run_env_loop(num_game):
    env = rl_env.HanabiEnv({"players": 2, "seed": 1})
    num_step = 0

    for _ in range(num_game):
        num_step += run_env(env)
        # num_step += env.num_step

    return num_step


def run_env_loop_multithread(num_game_per_thread, num_thread):
    futs = []
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        for _ in range(num_thread):
            futs.append(executor.submit(lambda: run_env_loop(num_game_per_thread)))

    steps = [f.result() for f in futs]
    return np.sum(steps)


if __name__ == "__main__":
    import sys
    print(f"nogil={getattr(sys.flags, 'nogil', False)}")

    for num_t in [1, 2, 4, 8, 16, 32]:
        # print(num_t)
        t = time.time()
        num_step = run_env_loop_multithread(100, num_t)
        end_t = time.time()
        print(f"{num_t} thread {num_step / (end_t - t):.2f} steps/s")
