import set_path
import torch
import hle


class HanabiEnv:
    def __init__(self, *, num_player, seed, max_len, bomb=0, start_player=-1):
        self.num_player = num_player
        self.seed = seed
        self.max_len = max_len
        self.start_player = start_player
        if self.start_player >= 0:
            random_start_player = 0
        else:
            random_start_player = 1
        params = {
            "players": str(self.num_player),
            "seed": str(seed),
            "bomb": str(bomb),
            "random_start_player": str(random_start_player)
        }
        # print("before creating game")
        self.hle_game = hle.HanabiGame(params)
        # print("after creating game")
        self.hle_state = None
        self.reward = 0

        self.num_step = 0

    def observation_dim(self):
        """return (priv_dim, publ_dim)"""
        encoder = hle.ObservationEncoder(self.hle_game)
        dim = encoder.shape()[0]
        priv_dim = dim - self.hle_game.num_bits_per_hand()
        publ_dim = priv_dim - self.hle_game.num_bits_per_hand()
        return priv_dim, publ_dim

    def num_action(self):
        return self.hle_game.max_moves() + 1  # +1 for the no-op move

    def no_op_action(self):
        return self.hle_game.max_moves()

    def reset(self):
        # print("before creating state")
        self.hle_state = hle.HanabiState(self.hle_game, self.start_player)
        # print("after creating state")
        while self.hle_state.is_chance_player():
            self.hle_state.apply_random_chance()
        self.num_step = 0

    def cur_player(self):
        assert self.hle_state is not None
        player = self.hle_state.cur_player()
        assert player >= 0
        return player

    def get_move(self, move_idx):
        return self.hle_game.get_move(move_idx)

    def observe(self, player_idx):
        # print("observe", self.num_step)
        game = self.hle_game
        state = self.hle_state

        obs = hle.HanabiObservation(state, player_idx, False)
        encoder = hle.ObservationEncoder(game)
        s = encoder.encode(obs, False, [], False, [], [], False)
        priv_s = s[game.num_bits_per_hand():]
        publ_s = priv_s[game.num_bits_per_hand():]

        legal_moves = obs.legal_moves()
        vec_legal_move = [0 for _ in range(1 + game.max_moves())]
        for legal_move in legal_moves:
            vec_legal_move[game.get_move_uid(legal_move)] = 1
        if len(legal_moves) == 0:
            vec_legal_move[-1] = 1

        feats = {
            "priv_s": torch.tensor(priv_s),
            "publ_s": torch.tensor(publ_s),
            "legal_move": torch.tensor(vec_legal_move)
        }
        return feats

    def is_null(self):
        return self.hle_state is None

    def terminated(self):
        if self.hle_state is None:
            return True
        if self.hle_state.is_terminal():
            return True
        if self.max_len > 0 and self.num_step >= self.max_len:
            return True
        return False

    def step(self, move: hle.HanabiMove):
        assert self.hle_state is not None, "did you forget to reset?"
        prev_score = self.hle_state.score()

        self.hle_state.apply_move(move)
        self.num_step += 1

        if not self.hle_state.is_terminal():
            while self.hle_state.is_chance_player():
                self.hle_state.apply_random_chance()

        self.reward = self.hle_state.score() - prev_score

    def score(self):
        assert self.hle_state is not None
        return self.hle_state.score()


if __name__ == "__main__":
    env = HanabiEnv(num_player=2, seed=1, max_len=-1, bomb=1)
    env.reset()
    obs = env.observe(0)
    for k, v in obs.items():
        print(k, v.size())

    while not env.terminated():
        print(env.num_step)
        move = env.hle_game.get_move(5)
        env.step(move)
    print(f"terminted after: {env.num_step} steps, score {env.score()}")
