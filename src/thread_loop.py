from env import HanabiEnv
from r2d2_actor import R2D2Actor
import r2d2_model
import utils


class ThreadLoop:
    def __init__(self, envs, actors, num_game):
        assert len(envs) == len(actors)
        assert num_game < 0 or num_game >= len(envs)

        self.envs = envs
        self.env_actors = actors
        # if num_game < 0, it will be infinite loop
        self.num_game = num_game
        self.scores = []
        self.num_step = 0

    @utils.kill_all_on_failure
    def loop(self):
        env_running = [True for _ in self.envs]
        num_env_running = len(self.envs)
        while num_env_running > 0:
            for i, (env, actors) in enumerate(zip(self.envs, self.env_actors)):
                if not env_running[i]:
                    continue

                if env.terminated():
                    assert self.num_game > 0
                    self.num_game -= 1

                    env.reset()
                    for actor in actors:
                        actor.init()

                for actor in actors:
                    actor.observe(env)

            for i, (env, actors) in enumerate(zip(self.envs, self.env_actors)):
                if not env_running[i]:
                    continue

                actions = [actor.decide_action() for actor in actors]
                move = env.get_move(actions[env.cur_player()])
                env.step(move)

                if env.terminated():
                    self.num_step += env.num_step
                    self.scores.append(env.score())
                    if self.num_game == 0:
                        env_running[i] = False
                        num_env_running -= 1


def create_envs_and_actors(num_env, num_player, seed, max_len, model_wrapper):
    envs = []
    env_actors = []
    for i in range(num_env):
        env = HanabiEnv(num_player=num_player, seed=seed + num_player * i, max_len=max_len)
        envs.append(env)
        actors = [R2D2Actor(j, seed + i * num_player + j, model_wrapper) for j in range(num_player)]
        env_actors.append(actors)

    return envs, env_actors


class DummyActor:
    def __init__(self, player_idx, seed):
        self.player_idx = player_idx
        self.seed = seed
        self.action = -1

    def init(self):
        pass

    def observe(self, env: HanabiEnv):
        obs = env.observe(self.player_idx)
        if env.hle_state.cur_player() == self.player_idx:
            legal_move = obs["legal_move"]
            legal_move[5:10] = 0
            self.action = legal_move.float().multinomial(1).item()
        else:
            self.action = -1

    def decide_action(self):
        return self.action


def create_envs_and_dummy_actors(num_env, num_player, seed, max_len):
    envs = []
    env_actors = []
    for i in range(num_env):
        env = HanabiEnv(num_player=num_player, seed=seed + num_player * i, max_len=max_len)
        envs.append(env)
        actors = [DummyActor(j, seed + i * num_player + j) for j in range(num_player)]
        env_actors.append(actors)

    return envs, env_actors


@utils.kill_all_on_failure
def test(num_thread, game_per_thread):
    print(f"running with {num_thread} threads and {game_per_thread} games in each thread")

    model = r2d2_model.load_default_model("cuda")
    model_wrapper = BatchedModelWrapper(model, True)
    model_wrapper.register_method("act", 4000)
    model_wrapper.start()

    t = time.time()
    thread_loops = []
    for i in range(num_thread):
        envs, actors = create_envs_and_actors(game_per_thread, 2, i * game_per_thread, -1, model_wrapper)
        thread_loop = ThreadLoop(envs, actors, 2000)
        thread_loops.append(thread_loop)
    print(f"time to create env: {time.time() - t:.5f}")
    threads = []
    t = time.time()
    for tl in thread_loops:
        threads.append(threading.Thread(target=tl.loop, args=()))
        threads[-1].start()

    for thread in threads:
        thread.join()

    run_time = time.time() - t
    print(f"num_thread: {num_thread}, game_per_thread: {game_per_thread}")
    total_num_game = sum([len(tl.scores) for tl in thread_loops])
    print(f"time to run games: {run_time:.2f}, speed: {total_num_game / run_time:.2f} game/s")
    total_num_step = sum([tl.num_step for tl in thread_loops])
    print(f"speed: {total_num_step / run_time:.2f} step/s")

    mean_batchsize = np.mean(model_wrapper._actual_batchsizes["act"])
    print(f"mean batchsize: {mean_batchsize:.2f}")

    model_wrapper.stop()

    scores = []
    for tl in thread_loops:
        scores.extend(tl.scores)
    print(f"avg score: {np.mean(scores)}")
    return


@utils.kill_all_on_failure
def test_with_dummy_actor(num_thread, game_per_thread):
    print(f"running with {num_thread} threads and {game_per_thread} games in each thread")

    t = time.time()
    thread_loops = []
    for _ in range(num_thread):
        envs, actors = create_envs_and_dummy_actors(game_per_thread, 2, 1, -1)
        thread_loop = ThreadLoop(envs, actors, 2000)
        thread_loops.append(thread_loop)
    print(f"time to create env: {time.time() - t:.5f}")
    threads = []
    t = time.time()
    for tl in thread_loops:
        threads.append(threading.Thread(target=tl.loop, args=()))
        threads[-1].start()

    for thread in threads:
        thread.join()

    run_time = time.time() - t
    print(f"num_thread: {num_thread}, game_per_thread: {game_per_thread}")
    total_num_games = sum([len(tl.scores) for tl in thread_loops])
    print(f"time to run games: {run_time:.2f}, speed: {total_num_games / run_time:.2f} game/s")

    total_num_step = sum([tl.num_step for tl in thread_loops])
    print(f"speed: {total_num_step / run_time:.2f} step/s")


if __name__ == "__main__":
    import time
    import threading

    import numpy as np
    from batched_model_wrapper import BatchedModelWrapper

    # test_with_dummy_actor(1, 100)
    # test_with_dummy_actor(16, 100)
    # test_with_dummy_actor(4, 100)

    # test(1, 100)
    # test(2, 100)
    # test(4, 100)
    # test(6, 100)
    # test(12, 100)
    test(16, 100)
    # test(80, 200)
