import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import src.option as option
from src.dataset import Dataset
from src.model import Model
from src.test import test
from src.utils import VisdomLinePlotter, norm_features, select_rgb_list


def test_inference_time(dataloader, model, device):
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    avg_inference_time = []

    # WARM UP
    dummy_input = torch.randn(1, 10, 1, 1024, dtype=torch.float).to(device)
    for _ in range(10):
        model(inputs=dummy_input)

    # random input for testing inference I3D feature
    input = torch.randn(1, 10, np.random.randint(64, 6250), 1024, dtype=torch.float).to(
        device
    )

    with torch.no_grad():
        model.eval()

        input = next(iter(dataloader))
        input = input.to(device)  # B x T x 10 x D
        input = input.permute(0, 2, 1, 3)  # B x 10 x T x D

        if input.size()[1] == 1:
            input = input.permute(1, 0, 2, 3)  # 1 x B x T x D

        input = input[:, :, :2, :]

        # repeat the process 30 times
        for i in range(30):
            starter.record()

            model(inputs=input)

            ender.record()

            # WAIT FOR GPU SYNC
            if device.type == "cuda":
                torch.cuda.synchronize()

            curr_time = starter.elapsed_time(ender)

            avg_inference_time.append((curr_time / input.shape[2]))

    X = np.asarray(avg_inference_time)
    print(X.mean(), X.std())


def plot_gt(file_name, feature_path, gt_path, args):
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


if __name__ == "__main__":
    args = option.parser.parse_args()

    viz = VisdomLinePlotter(env_name=args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SHT, UCF features are stored as a single .npy file for all the 10 crops (for a single video, shape: T x 10 x 1024)
    # XDV features are stored as a single .npy file for each of the 5 crop (for a single video, shape: T x 5 x 1024)

    test_loader = DataLoader(
        Dataset(args, list=select_rgb_list(args.dataset, test_mode=True), mode="test"),
        batch_size=(5 if args.dataset == "xdv" else 1),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    sap = "sap" in args.version
    multitask = "multitask" in args.version

    model = Model(
        args.feature_size,
        args.batch_size,
        args.k,
        args.m,
        args.sap_hidden_size,
        sap,
        multitask,
    )

    checkpoint = torch.load(args.pretrained_ckpt)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    clip_auc, clip_ap, _, _ = test(
        test_loader, model, args, viz, device, -1, print_metrics=True
    )

    test_inference_time(test_loader, model, device)
