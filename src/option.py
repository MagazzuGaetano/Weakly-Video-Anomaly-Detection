import argparse

parser = argparse.ArgumentParser(description="RTFM")

parser.add_argument(
    "--dataset", default="shanghai", help="dataset to train on (default: shanghai)"
)  # shanghai, ucf, xdv
parser.add_argument("--feat-extractor", default="i3d", choices=["i3d", "c3d"])
parser.add_argument(
    "--feature-size", type=int, default=1024, help="size of feature (default: 1024)"
)  # 1024, 4096
parser.add_argument(
    "--rgb-list",
    default="list/shanghai-i3d-train-10crop.list",
    help="list of rgb features",
)
parser.add_argument(
    "--test-rgb-list",
    default="list/shanghai-i3d-test-10crop.list",
    help="list of test rgb features",
)
parser.add_argument("--gt", default="list/gt-sh.npy", help="file of ground truth ")

parser.add_argument(
    "--vad_lr", type=float, default=1e-3, help="video anomaly detection learning rate"
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
    default=5000,
    help="maximum iteration to train (default: 100)",
)
parser.add_argument(
    "--m",
    type=int,
    default=150,
    help="margin considered in the Feature Magnitude Learning",
)
parser.add_argument(
    "--k", type=int, default=7, help="number of the top-k snippets considered"
)

parser.add_argument("--pretrained-ckpt", default=None, help="ckpt for pretrained model")
parser.add_argument(
    "--plot-freq", type=int, default=10, help="frequency of plotting (default: 10)"
)

parser.add_argument("--model-name", default="rtfm", help="name to save model")
parser.add_argument("--num-classes", type=int, default=1, help="number of class")
parser.add_argument("--gpus", default=1, type=int, choices=[0], help="gpus")
parser.add_argument("--workers", default=0, help="number of workers in dataloader")

parser.add_argument(
    "--version", type=str, default="snippets_multitask_sap", help="model version"
)  # snippets, csss, snippets_multitask_sap, clips_multitask_sap, clips_multitask, snippets_multitask_gap

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
    default=1e-2,
    help="the weight of the video classification loss",
)
parser.add_argument(
    "--n_sample_clips", type=int, default=64, help="number of clips to sample"
)
parser.add_argument(
    "--sap_hidden_size", type=int, default=256, help="sap hidden layer size"
)

parser.add_argument(
    "--L", type=int, default=32, help="number of subsets or clips to sample for CSSS"
)  # 16, 32
parser.add_argument(
    "--T", type=int, default=3, help="number of consecutive clips to use for CSSS"
)  # 3, 5, 7
