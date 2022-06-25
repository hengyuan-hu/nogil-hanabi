from r2d2_model import R2D2Model
from env import HanabiEnv


class R2D2Actor:
    def __init__(self, player_idx, seed, model_wrapper):
        self.player_idx = player_idx
        self.seed = seed
        self.model_wrapper = model_wrapper

        self.rnn_hid = None
        self.fut_reply = None

    def init(self):
        self.rnn_hid = self.model_wrapper.model.get_init_rnn_hid()

    def observe(self, env: HanabiEnv):
        obs = env.observe(self.player_idx)
        obs.update(self.rnn_hid)

        self.fut_reply = self.model_wrapper.async_call("act", obs)

    def decide_action(self):
        # print(f"player {self.player_idx} getting result")
        reply = self.fut_reply.get()
        for k in self.rnn_hid:
            self.rnn_hid[k] = reply.pop(k)
        action = reply["a"].item()
        return action
