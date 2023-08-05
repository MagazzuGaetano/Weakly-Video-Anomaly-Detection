import argparse


parser = argparse.ArgumentParser(description="RTFM")
parser.add_argument(
    "--dataset", default="shanghai", help="dataset to train on (default: )"
)  # shanghai, ucf, xdv
parser.add_argument("--feat-extractor", default="i3d", choices=["i3d", "c3d"])
parser.add_argument(
    "--feature-size", type=int, default=1024, help="size of feature (default: 1024)"
)  # 1024, 4096
parser.add_argument(
    "--vad_lr", type=float, default=1e-4, help="video anomaly detection learning rate"
)
parser.add_argument(
    "--vad_weight_decay",
    type=float,
    default=5e-4,
    help="video anomaly detection weight decay",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="number of instances in a batch of data (default: 16)",
)
parser.add_argument(
    "--max-epoch",
    type=int,
    default=15000,
    help="maximum iteration to train (default: 100)",
)

parser.add_argument("--pretrained-ckpt", default="", help="ckpt for pretrained model")
parser.add_argument(
    "--plot-freq", type=int, default=10, help="frequency of plotting (default: 10)"
)

parser.add_argument("--model-name", default="rtfm", help="name to save model")
parser.add_argument("--workers", default=0, help="number of workers in dataloader")


parser.add_argument(
    "--version", type=str, default="sample_SAP", help="model version"
)  # snippets, sample_SAP, CSSS

parser.add_argument(
    "--vc_lr", type=float, default=1e-4, help="video classification learning rate"
)
parser.add_argument(
    "--vc_weight_decay",
    type=float,
    default=5e-4,
    help="video classification weight decay",
)
parser.add_argument(
    "--vc_alpha",
    type=int,
    default=1e-4,
    help="the weight of the video classification loss",
)
parser.add_argument(
    "--n_sample_clips",
    type=int,
    default=64,
    help="[only if sample_SAP version] - number of clips to sample",
)
parser.add_argument(
    "--sap_hidden_size",
    type=int,
    default=256,
    help="[only if sample_SAP version] - sap hidden layer size",
)

parser.add_argument(
    "--L",
    type=int,
    default=32,
    help="[only if CSSS version] - number of subsets or clips to sample for CSSS",
)  # 16, 32
parser.add_argument(
    "--T",
    type=int,
    default=3,
    help="[only if CSSS version] - number of consecutive clips to use for CSSS",
)  # 3, 5, 7


# python3 main.py --dataset 'shanghai'

# python3 test_only.py --dataset 'shanghai'

# python3 -m visdom.server

# RTFM paper hyperparameters
# - 0,001 initial LR [shanghai-tech, ucf-crime], 0,0001 initial LR [xd-violence],
# - 0.0005 weight decay, 64 batch size, 50 epochs
