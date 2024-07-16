import torch
from torch.utils.data import DataLoader

import src.option as option
from src.dataset import Dataset
from src.model import Model
from src.test import test, test_inference_time
from src.utils import VisdomLinePlotter, select_rgb_list


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

    mean, std = test_inference_time(test_loader, model, device)
    print("Inference Time: {} ± {} ms".format(mean, std))
