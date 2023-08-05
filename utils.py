import visdom
import numpy as np
import torch
import random
import os

from visdom import Visdom


def random_extract(feat, t_max):
    r = np.random.randint(len(feat) - t_max)
    return feat[r : r + t_max]


def uniform_extract(feat, t_max):
    r = np.linspace(0, len(feat) - 1, t_max, dtype=np.uint16)
    return feat[r, :]


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
        pad_width = (
            (0, min_len - np.shape(feat)[0]),
            (0, 0),
            (0, 0),
        )
        return np.pad(feat, pad_width, mode="constant", constant_values=0)
    else:
        return feat


def process_feat_2(feat, length, is_random=True):
    feat = feat.transpose(1, 0, 2)
    if len(feat) > length:
        if is_random:
            out_feat = random_extract(feat, length)
        else:
            out_feat = uniform_extract(feat, length)
    else:
        out_feat = pad(feat, length)
    out_feat = out_feat.transpose(1, 0, 2)
    return out_feat


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name="main"):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel="Epochs",
                    ylabel=var_name,
                ),
            )
        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update="append",
            )


class Visualizer(object):
    def __init__(self, env="default", **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        """
        self.plot('loss', 1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]),
            X=np.array([x]),
            win=str(name),
            opts=dict(title=name),
            update=None if x == 0 else "append",
            **kwargs
        )
        self.index[name] = x + 1

    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))

    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)

    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)


def norm_features(feat):
    # single clip features normalized (L2)
    # for i in range(feat.shape[0]):
    #    feat[i,:,:] = feat[i,:,:] / np.linalg.norm(feat[i,:,:])
    # feat = feat / np.linalg.norm(feat) # worked for I3D and C3D(SHT only)
    # feat = feat / np.linalg.norm(feat, axis=-1, keepdims=True) # per C3D (UCF e XDV)
    # print(feat.min(), feat.max())
    return feat


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int32)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


def sample_m_clips(features, m=32):
    _, nclips, _ = features.shape  # (ncrops, nclips, D)

    sampled_features = features

    if nclips >= m:
        # r = np.arange(0, nclips)
        # sample_indices = np.random.choice(r, size=(m), replace=False)
        # sample_indices = np.sort(sample_indices)
        sample_indices = np.linspace(0, nclips - 1, m, dtype=int)
        sampled_features = features[:, sample_indices, :]
    else:
        # loop del video
        sample_indices = np.tile(np.arange(0, nclips), (m // nclips + 1))
        # clip to m clips
        sample_indices = sample_indices[:m]

        sampled_features = features[:, sample_indices, :]

    return sampled_features


def sample_subsets_special(features, L=32, T=3):
    # implementazione Continuos Sparse Sampling Strategy copiata da https://github.com/shengyangsun/LSTC_VAD
    ncrops, nclips, feat_dim = features.shape  # (ncrops, nclips, D)

    kk = nclips - T

    if kk // (L + 1) < 1:
        move = 0
    else:
        move = np.random.randint(kk // (L + 1))

    if kk < 0:
        chosen = np.linspace(0, nclips - 1, num=L + 1, dtype=int).repeat(T)
        chosen = chosen.repeat(T).reshape([-1, T])
    else:
        chosen = np.linspace(0, kk, num=L + 1, dtype=int) + move
        chosen = chosen.repeat(T).reshape([-1, T]) + np.arange(0, T, 1, dtype=int)
    chosen = chosen.reshape([-1])
    divided_features = features[:, chosen[: L * T], :]
    divided_features = divided_features.reshape(
        [ncrops, L, T, feat_dim]
    )  # [10, L, T, F]
    divided_features = divided_features.reshape(
        [ncrops, L * T, feat_dim]
    )  # [10, L * T, F]
    return divided_features


def set_seeds(seed):
    print("set seed {}".format(seed))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark=False
    # torch.backends.cudnn.deterministic=True


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=0)[0])
        min_val = relu(torch.min(act_map, dim=0)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def modelsize(model, input, type_size=4):
    # check GPU utilisation
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print(
        "Model {} : params: {:4f}M".format(
            model._get_name(), para * type_size / 1000 / 1000
        )
    )

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print(
        "Model {} : intermedite variables: {:3f} M (without backward)".format(
            model._get_name(), total_nums * type_size / 1000 / 1000
        )
    )
    print(
        "Model {} : intermedite variables: {:3f} M (with backward)".format(
            model._get_name(), total_nums * type_size * 2 / 1000 / 1000
        )
    )


def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(str(test_info["test_AUC"][-1]))
    fo.close()
