from dataclasses import dataclass
import torch
from torch import nn


def duel(v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor) -> torch.Tensor:
    assert a.size() == legal_move.size()
    assert legal_move.dim() == 3  # seq, batch, dim
    legal_a = a * legal_move
    q = v + legal_a# - legal_a.mean(2, keepdim=True)
    return q


def cross_entropy(net, lstm_o, target_p, hand_slot_mask, seq_len):
    # target_p: [seq_len, batch, num_player, 5, 3]
    # hand_slot_mask: [seq_len, batch, num_player, 5]
    logit = net(lstm_o).view(target_p.size())
    q = nn.functional.softmax(logit, -1)
    logq = nn.functional.log_softmax(logit, -1)
    plogq = (target_p * logq).sum(-1)
    xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(min=1e-6)

    if xent.dim() == 3:
        # [seq, batch, num_player]
        xent = xent.mean(2)

    # save before sum out
    seq_xent = xent
    xent = xent.sum(0)
    assert xent.size() == seq_len.size()
    avg_xent = (xent / seq_len).mean().item()
    return xent, avg_xent, q, seq_xent.detach()


@dataclass
class LSTMNetConfig:
    in_dim: int
    hid_dim: int
    out_dim: int
    num_lstm_layer: int


class LSTMNet(nn.Module):
    def __init__(self, config: LSTMNetConfig):
        super().__init__()
        self.in_dim = config.in_dim
        self.hid_dim = config.hid_dim
        self.out_dim = config.out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = config.num_lstm_layer

        ff_layers = [nn.Linear(self.in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(self.hid_dim, self.hid_dim, self.num_lstm_layer)
        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

    def get_init_rnn_hid(self):
        """
        Get inital h for the single agent/player,
        batchsize dimension is omitted
        """
        shape = (self.num_lstm_layer, self.hid_dim)
        hid = {"h": torch.zeros(*shape), "c": torch.zeros(*shape)}
        return hid

    def adv(self, obs, hid):
        """
        Compute advantage for observation of one step
        obs contains:
            priv_s: [1, batchsize, priv_dim]
            legal_action: [1, batchsize, num_action]
        hid contains:
            h: [num_layer, batchsize, rnn_dim]
            c0: [num_layer, batchsize, rnn_dim]
        """
        priv_s = obs["priv_s"]
        assert priv_s.dim() == 3, "must be [seq_len, batch, priv_dim]"
        x = self.net(priv_s)
        if hid is None:
            o, (h, c) = self.lstm(x)
        else:
            # print(list(hid.keys()))
            o, (h, c) = self.lstm(x, (hid["h"], hid["c"]))
        new_hid = {"h": h, "c": c}
        a = self.fc_a(o)
        return a, new_hid

    def forward(self, obs, hid, action):
        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]
        assert priv_s.dim() == 3 # must be [seq_len, batch, priv_dim]

        x = self.net(priv_s)
        o, _ = self.lstm(x, (hid["h"], hid["c"]))
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)
        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        # -> qa: [seq_len, batch], i.e. q(a)
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()
        return qa, greedy_action, q, o
