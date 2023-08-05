from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import VisdomLinePlotter, sample_m_clips, save_best_record, set_seeds
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
import gc
import random
import numpy as np
import os
from sklearn.model_selection import ParameterGrid


if __name__ == "__main__":
    import time

    start_time = time.time()

    args = option.parser.parse_args()

    global viz
    viz = VisdomLinePlotter(env_name=args.dataset)

    # params = {
    #   'vad_lr': [1e-3, 1e-4],
    #   'vad_weight_decay': [5e-3, 5e-4],
    #   'batch_size': [32],
    #   'vc_alpha': [1e-3, 1e-4],
    #   'vc_lr': [1e-3, 1e-4],
    #   'vc_weight_decay': [5e-3, 5e-4],
    #   'epochs' : [100],
    #   'n_sample_clips': [32, 64, 128],
    #   'sap_hidden_size': [512, 256, 128],
    # }

    params = {}
    param_grid = ParameterGrid(params)

    for param in param_grid:
        print(param)

    TUNING = len(params.keys()) > 0

    metric = "AUC"  # 'AUC', 'AP'
    best_config_metric = -1
    best_config = {}

    i = 0
    for dict_ in param_grid:

        # clear cache
        torch.cuda.empty_cache()

        # sht:0, ucf:4, xdv: 1(c3d), xdv: 2(i3d)
        SEED = 0  # dict_['seed']
        set_seeds(SEED)

        # override args parameters with current configuration
        if TUNING:
            args.vad_lr = dict_["vad_lr"]
            args.vad_weight_decay = dict_["vad_weight_decay"]
            args.batch_size = dict_["batch_size"]
            args.max_epoch = dict_["epochs"]

            args.vc_lr = dict_["vc_lr"]
            args.vc_weight_decay = dict_["vc_weight_decay"]
            args.vc_alpha = dict_["vc_alpha"]
            args.n_sample_clips = dict_["n_sample_clips"]
            args.sap_hidden_size = dict_["sap_hidden_size"]

        def worker_init(worked_id):
            np.random.seed(SEED + worked_id)
            random.seed(SEED + worked_id)

        test_loader = DataLoader(
            Dataset(args, test_mode=True),
            batch_size=(5 if args.dataset == "xdv" else 1),
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
        )
        print("test loader", len(test_loader))

        train_nloader = DataLoader(
            Dataset(args, test_mode=False, is_normal=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            worker_init_fn=worker_init,
            pin_memory=False,
            drop_last=True,
        )
        print("train normal loader", len(train_nloader))

        train_aloader = DataLoader(
            Dataset(args, test_mode=False, is_normal=False),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            worker_init_fn=worker_init,
            pin_memory=False,
            drop_last=True,
        )
        print("train abnormal loader", len(train_aloader))

        use_sap = args.version == "sample_SAP"
        model = Model(args.feature_size, args.batch_size, args.sap_hidden_size, use_sap)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if not os.path.exists("./ckpt"):
            os.makedirs("./ckpt")

        optimizer = optim.Adam(
            [
                {"params": model.params.temporal_features.parameters()},
                {"params": model.params.snippet_classifier.parameters()},
                {
                    "params": model.params.video_classifier.parameters(),
                    "lr": args.vc_lr,
                    "weight_decay": args.vc_weight_decay,
                },
            ],
            lr=args.vad_lr,
            weight_decay=args.vad_weight_decay,
        )

        # continue training
        checkpoint_path = args.pretrained_ckpt
        ckpt_step = -1
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            ckpt_step = checkpoint["epoch"]
            loadern_iter = iter(train_nloader)
            loadera_iter = iter(train_aloader)

        test_info = {"epoch": [], "test_AUC": [], "test_AP": []}
        best_AUC = -1
        best_AP = -1
        output_path = "./output"  # put your own path here
        auc, ap = test(
            test_loader, model, args, viz, device, 0, print_metrics=not TUNING
        )

        for step in tqdm(
            range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True
        ):
            if step <= ckpt_step:
                continue

            if (step - 1) % len(train_nloader) == 0:
                loadern_iter = iter(train_nloader)

            if (step - 1) % len(train_aloader) == 0:
                loadera_iter = iter(train_aloader)

            train(
                loadern_iter,
                loadera_iter,
                model,
                args.batch_size,
                optimizer,
                viz,
                device,
                step,
                args.version,
                args.vc_alpha,
                print_metrics=not TUNING,
            )

            if step % 5 == 0 and step >= 50:

                auc, ap = test(
                    test_loader,
                    model,
                    args,
                    viz,
                    device,
                    step,
                    print_metrics=not TUNING,
                )
                test_info["epoch"].append(step)
                test_info["test_AUC"].append(auc)
                test_info["test_AP"].append(ap)

                if test_info["test_AUC"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC"][-1]

                if test_info["test_AP"][-1] > best_AP:
                    best_AP = test_info["test_AP"][-1]

                    # if not TUNING:
                    #   torch.save({
                    #     'epoch': step,
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     }, './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
                    #   save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))

        # if not TUNING:
        #   torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')

        best_metric = best_AUC if metric == "AUC" else best_AP

        if best_metric > best_config_metric:
            best_config_metric = best_metric
            best_config = dict_

        i += 1

        print(
            "-----------------------------------------------------------------------------------------"
        )
        print("Best Metric({}): {}".format(metric, best_metric))
        print(
            "-----------------------------------------------------------------------------------------"
        )

        if TUNING:
            print("Curr config: {}".format(dict_))
            print("Runs: {} / {}".format(i, len(param_grid)))
            print(
                "-----------------------------------------------------------------------------------------"
            )

    print(
        "-----------------------------------------------------------------------------------------"
    )
    print("Best config: {}".format(best_config))
    print("Best Metric({}): {}".format(metric, best_config_metric))
    print(
        "-----------------------------------------------------------------------------------------"
    )
    print("--- %s seconds ---" % (time.time() - start_time))
