from typing import Dict
from dataclasses import dataclass
import torch
from torch import nn

from env import HanabiEnv
from net import LSTMNet, LSTMNetConfig, PublicLSTMNet, PublicLSTMNetConfig
import utils


@dataclass
class AlgoConfig:
    iql: bool
    vdn: bool
    # TODO: support more
    # boltzmann_act: bool
    # maxent: bool
    # aux: bool
    # obl: bool

    def validate(self):
        assert self.iql or self.vdn
        assert not (self.iql and self.vdn)
        # if self.maxent:
        #     assert self.boltzmann_act


@dataclass
class QLearningConfig:
    multi_step: int
    gamma: float
    eta: float
    uniform_priority: bool


class R2D2Model(nn.Module):
    """A model is the outer-most interface for all neural network logics"""
    def __init__(
        self,
        device,
        net_class,
        net_config,
        algo_config,
        q_learning_config,
    ):
        super().__init__()
        self.device = device
        self.online_net = net_class(net_config)
        self.target_net = net_class(net_config)

        self.online_net.to(device)
        self.target_net.to(device)
        self.target_net.train(False)

        for p in self.target_net.parameters():
            p.requires_grad = False
        self.target_net.train(False)

        self.algo_config = algo_config
        self.q_learning_config = q_learning_config
        self.hid_keys = list(self.get_init_rnn_hid().keys())

    def to(self, device):
        self.device = device
        nn.Module.to(self, device)

    def get_init_rnn_hid(self) -> Dict[str, torch.Tensor]:
        return self.online_net.get_init_rnn_hid()

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def act(self, input_: Dict[str, torch.Tensor]):
        """
        Acts for 1 timestep on the given obs, and hidden states
        obs contains:
            priv_s: [batchsize, priv_dim]
            publ_s: [batchsize, publ_dim]
            legal_action: [batchsize, num_action]
            eps (optional): [batchsize]
        hid contains:
            h0: [batchsize, num_layer, rnn_dim]
            c0: [batchsize, num_layer, rnn_dim]

        output:
        reply contains:
            a: [batchsize]
        new_hid contains the same variables of the same shape as in hid
        """
        obs = {}
        hid = {}
        for k, v in input_.items():
            if k in self.hid_keys:
                hid[k] = v.to(self.device)
            else:
                obs[k] = v.to(self.device)

        assert obs["priv_s"].dim() == 2
        reply, new_hid = self.eps_greedy(obs, hid)

        reply.update(new_hid)
        reply = utils.detach_and_to_device(reply, "cpu")
        return reply

    def eps_greedy(self, obs: Dict[str, torch.Tensor], hid: Dict[str, torch.Tensor]):
        batchsize = obs["priv_s"].size(0)
        legal_move = obs.pop("legal_move").float()
        if "eps" in obs:
            eps = obs.pop("eps")
            assert eps.dim() == 1
        else:
            eps = torch.zeros(batchsize, device=legal_move.device)

        # convert the shapes of inputs to the ones needed by online_net
        for k, v in obs.items():
            obs[k] = v.unsqueeze(0)

        for k, v in hid.items():
            hid[k] = v.transpose(0, 1).contiguous()

        adv, new_hid = self.online_net.adv(obs, hid)
        for k, v in new_hid.items():
            new_hid[k] = v.transpose(0, 1).contiguous()

        legal_adv = adv.squeeze(0) - (1 - legal_move) * 1e30
        greedy_action = legal_adv.argmax(1)
        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).detach().long()
        return {"a": action}, new_hid


def load_default_model(device):
    num_player = 2
    env = HanabiEnv(num_player=num_player, seed=1, max_len=-1, bomb=1)
    priv_dim, publ_dim = env.observation_dim()
    num_action = env.num_action()
    algo_config = AlgoConfig(True, False)
    q_config = QLearningConfig(1, 0.999, 0.9, True)

    weight_file = "/home/hhu/dev/hanabi/adv-models/monster-bot/model0.pthw"
    d = torch.load(weight_file)
    if "priv_net.0.weight" in list(d.keys()):
        net_config = PublicLSTMNetConfig(priv_dim, publ_dim, 512, num_action, 2)
        net_cls = PublicLSTMNet
    else:
        net_config = LSTMNetConfig(priv_dim, 512, num_action, 2)
        net_cls = LSTMNet
    r2d2_model = R2D2Model("cpu", net_cls, net_config, algo_config, q_config)
    utils.load_weight(r2d2_model.online_net, weight_file, "cpu")
    r2d2_model.to(device)
    return r2d2_model
