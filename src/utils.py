import visdom
import numpy as np
import torch
import random
import os

from visdom import Visdom
from sklearn.metrics import auc, roc_curve, average_precision_score


def norm_features(feat, args):
    # single clip features normalized (L2)
    if args.dataset == "shanghai":
        feat = feat / np.linalg.norm(feat)
    elif args.dataset == "ucf" or args.dataset == "xdv":
        feat = feat / np.linalg.norm(feat, axis=-1, keepdims=True)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    return feat


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


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int32)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]

    return new_feat


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


def compute_metrics(gt, pred, print_metrics=True):
    fpr, tpr, th = roc_curve(gt, pred)
    rec_auc = auc(fpr, tpr)

    if print_metrics:
        print("auc : " + str(rec_auc))

    ap = average_precision_score(gt, pred, pos_label=1)

    if print_metrics:
        print("ap : " + str(ap))

    return rec_auc, ap


def select_rgb_list(dataset, test_mode=True, is_normal=True):
    out_list = []
    rgb_list_file = ""

    if dataset == "shanghai":
        if test_mode:
            rgb_list_file = "list/shanghai-i3d-test-10crop.list"
        else:
            rgb_list_file = "list/shanghai-i3d-train-10crop.list"

    elif dataset == "ucf":
        if test_mode:
            rgb_list_file = "list/ucf-i3d-test.list"
        else:
            rgb_list_file = "list/ucf-i3d.list"

    elif dataset == "xdv":
        if test_mode:
            rgb_list_file = "list/xdv_rgb_test.list"
        else:
            rgb_list_file = "list/xdv_rgb.list"

    # read the rgb list
    with open(rgb_list_file) as f:
        rgb_list_file = f.readlines()
        out_list = rgb_list_file

    if test_mode is False:
        if dataset == "shanghai":
            if is_normal:
                out_list = rgb_list_file[63:]
            else:
                out_list = rgb_list_file[:63]

        elif dataset == "ucf":
            if is_normal:
                out_list = rgb_list_file[810:]
            else:
                out_list = rgb_list_file[:810]

        elif dataset == "xdv":
            if is_normal:
                out_list = [f for f in rgb_list_file if "_label_A" in f]
                out_list = [
                    f for f in out_list if "v=8cTqh9tMz_I__#1_label_A" not in f
                ]  # discard corrupted files
            else:
                out_list = [f for f in rgb_list_file if "_label_A" not in f]

        print(len(out_list))
        print("{} list for {}".format("normal" if is_normal else "abnormal", dataset))

    return out_list


def plot_gt(file_name, feature_path, gt_path, model, device, args):
    import matplotlib.pyplot as plt

    with torch.no_grad():
        model.eval()

        # read features
        if args.dataset == "xdv":
            single_crop_features = []
            for i in range(5):
                features = np.load(
                    os.path.join(
                        feature_path,
                        file_name.replace(".npy", "") + "__{}.npy".format(i),
                    ),
                    allow_pickle=True,
                )
                features = np.array(features, dtype=np.float32)
                features = norm_features(features)
                single_crop_features.append(features)

            features = np.asarray(single_crop_features)
        else:
            features = np.load(os.path.join(feature_path, file_name), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            features = norm_features(features)

        input = torch.from_numpy(features).unsqueeze(0)
        input = input.to(device)  # B x T x 10 x D

        if args.dataset != "xdv":
            input = input.permute(0, 2, 1, 3)  # B x 10 x T x D

        _, _, _, _, _, _, anomaly_scores, _, _, _, video_score = model(inputs=input)

        anomaly_scores = torch.squeeze(anomaly_scores, 1)
        anomaly_scores = torch.mean(anomaly_scores, 0)

        if "multitask" in args.version:
            video_score = torch.squeeze(video_score, 1)

        anomaly_scores = list(anomaly_scores.cpu().detach().numpy())
        anomaly_scores = np.repeat(np.array(anomaly_scores), 16)

        gt = np.load(os.path.join(gt_path, file_name))

        plt.plot(anomaly_scores, label="predicted scores")
        plt.plot(gt, label="gt scores")
        plt.ylabel("anomaly score")
        plt.title(file_name.replace(".npy", ".mp4"))
        plt.ylim((0, 1.1))
        plt.legend()
        plt.show()


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
            **kwargs,
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
