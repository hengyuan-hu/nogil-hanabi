import os
import signal
import traceback
import collections
import torch
import torch.multiprocessing as mp


global_ctx = None
def get_multiprocessing_ctx():
    """
    Get an instance of torch.multiprocessing that uses the "spawn"
    context method.  The instance returned is distinct from
    multiprocessing or torch.multiprocessing. This is to avoid
    clashing with poorly-written libraries or python dependencies that
    may set the start method of the global multiprocessing to
    something other than "spawn".
    """
    global global_ctx
    if global_ctx is None:
        global_ctx = mp.get_context("spawn")
    return global_ctx


def recursive_apply(obj, func):
    if torch.is_tensor(obj):
        return func(obj)
    if isinstance(obj, dict):
        ret = {}
        for k, v in obj.items():
            ret[k] = recursive_apply(v, func)
        return ret

    raise TypeError("Invalid type")


def to_device(obj, device):
    return recursive_apply(obj, lambda x: x.to(device))


def detach_and_to_device(obj, device):
    return recursive_apply(obj, lambda x: x.detach().to(device))


def load_weight(model, weight_file, device, *, state_dict=None):
    if state_dict is None:
        state_dict = torch.load(weight_file, map_location=device)
    source_state_dict = collections.OrderedDict()
    target_state_dict = model.state_dict()

    for k, v in target_state_dict.items():
        if k not in state_dict:
            print("warning: %s not loaded [not found in file]" % k)
            state_dict[k] = v
        elif state_dict[k].size() != v.size():
            print(
                "warnning: %s not loaded\n[size mismatch %s (in net) vs %s (in file)]"
                % (k, v.size(), state_dict[k].size())
            )
            state_dict[k] = v
    for k in state_dict:
        if k not in target_state_dict:
            print("removing: %s not used" % k)
        else:
            source_state_dict[k] = state_dict[k]

    model.load_state_dict(source_state_dict)


def kill_all_on_failure(func):
    def wrapped_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            traceback.print_exc()
            os.kill(os.getpid(), signal.SIGKILL)

    return wrapped_func


def load_default_model(device):
    from env import HanabiEnv
    from r2d2_model import R2D2Model, AlgoConfig, QLearningConfig
    from net import LSTMNet, LSTMNetConfig

    num_player = 2
    env = HanabiEnv(num_player=num_player, seed=1, max_len=-1, bomb=1)
    priv_dim, _ = env.observation_dim()
    num_action = env.num_action()
    net_config = LSTMNetConfig(priv_dim, 512, num_action, 2)
    algo_config = AlgoConfig(True, False)
    q_config = QLearningConfig(1, 0.999, 0.9, True)
    r2d2_model = R2D2Model("cpu", LSTMNet, net_config, algo_config, q_config)
    weight_file = "/checkpoint/hengyuan/hanabi_benchmark/vani_sad_aux_op/NETlstm_OP1_SEEDa/model0.pthw"
    d = torch.load(weight_file)
    load_weight(r2d2_model.online_net, weight_file, "cpu")
    r2d2_model.to(device)
    return r2d2_model
