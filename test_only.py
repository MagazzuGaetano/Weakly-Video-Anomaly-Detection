from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import VisdomLinePlotter, save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *


if __name__ == "__main__":
    args = option.parser.parse_args()
    config = Config(args)

    global viz
    viz = VisdomLinePlotter(env_name=args.dataset)

    test_loader = DataLoader(
        Dataset(args, test_mode=True),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = Model(args.feature_size, args.batch_size)
    checkpoint = torch.load("./ckpt/rtfm335-i3d.pkl")
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    auc = test(test_loader, model, args, viz, device, 0, print_metrics=False)
