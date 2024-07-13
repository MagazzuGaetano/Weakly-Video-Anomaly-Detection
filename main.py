import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.option as option
from src.dataset import Dataset
from src.model import Model
from src.test import test
from src.train import train
from src.utils import VisdomLinePlotter, select_rgb_list, set_seeds


def dataloader_setup(args):
    # read original train_list split into two list with class (normal/abnormal) video
    normal_train_list = select_rgb_list(args.dataset, test_mode=False, is_normal=True)
    abnormal_train_list = select_rgb_list(
        args.dataset, test_mode=False, is_normal=False
    )

    train_nloader = None
    train_aloader = None
    test_loader = None

    train_nloader = DataLoader(
        Dataset(args, list=normal_train_list, mode="train", is_normal=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        pin_memory=False,
        drop_last=True,
    )

    train_aloader = DataLoader(
        Dataset(args, list=abnormal_train_list, mode="train", is_normal=False),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        pin_memory=False,
        drop_last=True,
    )

    test_loader = DataLoader(
        Dataset(args, list=select_rgb_list(args.dataset, test_mode=True), mode="test"),
        batch_size=(5 if args.dataset == "xdv" else 1),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    print("train normal loader", len(train_nloader))
    print("train abnormal loader", len(train_aloader))

    if test_loader:
        print("test loader", len(test_loader))

    return train_aloader, train_nloader, test_loader


def save_checkpoint(step, model, optimizer, args, save_model):
    if save_model:
        torch.save(
            {
                "epoch": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "./ckpt/" + args.model_name + "{}-i3d.pkl".format(step),
        )


def update_tuning_records(df, curr_tuning_config, best_model_metrics):
    # add the current tuning configuration iperparameters to entry dataframe
    # add the best metrics obtained for this current tuning configuration to entry dataframe
    new_row_entry = pd.DataFrame.from_dict([curr_tuning_config | best_model_metrics])

    # append to df the new row entry
    df = pd.concat([df, new_row_entry], ignore_index=True)

    return df


if __name__ == "__main__":
    start_time = time.time()
    args = option.parser.parse_args()

    global viz
    viz = VisdomLinePlotter(env_name=args.dataset)

    # tuning UCF
    params = {
        "vad_lr": [1e-3],
        "vad_weight_decay": [5e-4],
        "batch_size": [32],
        "vc_alpha": [1e-3, 1e-4],
        "vc_lr": [1e-4, 1e-5],
        "vc_weight_decay": [5e-4, 5e-5],
        "n_sample_clips": [16, 32, 64],
        "sap_hidden_size": [512, 256, 128],
        "epochs": [500],
    }

    params = {}
    param_grid = ParameterGrid(params)

    for param in param_grid:
        print(param)

    train_mode = "training"
    print_metrics = False
    save_checkpoints = False

    if train_mode == "tuning":
        print_metrics = False
        save_checkpoints = False
    elif train_mode == "training":
        print_metrics = True
        save_checkpoints = True
    else:
        print_metrics = False
        save_checkpoints = True

    metric = "auc_clip"  # 'auc_clip', 'ap_clip', 'auc_video', 'ap_video'
    best_config_metric = -1
    best_config = {}

    plot_config = pd.DataFrame()
    csv_filename = "{}_{}_tuning-report.csv".format(args.feat_extractor, args.dataset)

    hyperparams = []
    # if a tuinig report already exist load it
    if os.path.isfile(csv_filename):
        plot_config = pd.read_csv(csv_filename, index_col=False)
        param_cols = ~plot_config.columns.isin(
            ["auc_clip", "ap_clip", "auc_video", "ap_video", "batch_size"]
        )

        # already explored configurations
        hyperparams = plot_config.loc[:, param_cols].to_dict(orient="records")

    print(len(param_grid))

    for i, dict_ in enumerate(param_grid):
        if train_mode == "tuning":
            dict_copy = dict_.copy()
            del dict_copy["batch_size"]

            # if the configuration was already explored skip it!
            if dict_copy in hyperparams:
                print("skip configuration {} / {}".format(i + 1, len(param_grid)))
                continue

        # clear cache
        torch.cuda.empty_cache()

        SEED = 0  # dict_['seed'] # sht:0, ucf:4, xdv:1(c3d), 2(i3d)
        set_seeds(SEED)

        # override args parameters with current configuration
        if train_mode == "tuning":
            args.vad_lr = dict_["vad_lr"]
            args.vad_weight_decay = dict_["vad_weight_decay"]
            args.batch_size = dict_["batch_size"]
            args.max_epoch = dict_["epochs"]
            args.m = dict_["m"]
            args.k = dict_["k"]

            args.vc_lr = dict_["vc_lr"]
            args.vc_weight_decay = dict_["vc_weight_decay"]
            args.vc_alpha = dict_["vc_alpha"]
            args.n_sample_clips = dict_["n_sample_clips"]
            args.sap_hidden_size = dict_["sap_hidden_size"]

        def worker_init(worked_id):
            np.random.seed(SEED + worked_id)
            random.seed(SEED + worked_id)

        train_aloader, train_nloader, test_loader = dataloader_setup(args)

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
        checkpoint_path = "./ckpt/rtfmX-i3d.pkl"
        ckpt_step = -1
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            ckpt_step = checkpoint["epoch"]
            loadern_iter = iter(train_nloader)
            loadera_iter = iter(train_aloader)

        best_model_metrics = {
            "auc_clip": -1,
            "ap_clip": -1,
            "auc_video": -1,
            "ap_video": -1,
        }
        output_path = "./output"  # put your own path here

        _, _, _, _ = test(
            test_loader, model, args, viz, device, 0, print_metrics=print_metrics
        )

        torch.cuda.empty_cache()

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
                print_metrics=print_metrics,
            )

            if step % 5 == 0 and step >= 50:
                auc_clip, ap_clip, auc_video, ap_video = test(
                    test_loader,
                    model,
                    args,
                    viz,
                    device,
                    step,
                    print_metrics=print_metrics,
                )

                # update the metrics of the best model (which is the one with the highest auc_clip)
                # and save the new best model
                if metric == "auc_clip":
                    if auc_clip > best_model_metrics[metric]:
                        best_model_metrics[metric] = auc_clip
                        best_model_metrics["ap_clip"] = ap_clip
                        best_model_metrics["auc_video"] = auc_video
                        best_model_metrics["ap_video"] = ap_video
                        save_checkpoint(step, model, optimizer, args, save_checkpoints)

                # update the metrics of the best model (which is the one with the highest ap_clip)
                # and save the new best model
                if metric == "ap_clip":
                    if ap_clip > best_model_metrics[metric]:
                        best_model_metrics["auc_clip"] = auc_clip
                        best_model_metrics[metric] = ap_clip
                        best_model_metrics["auc_video"] = auc_video
                        best_model_metrics["ap_video"] = ap_video
                        save_checkpoint(step, model, optimizer, args, save_checkpoints)

        if save_checkpoints:
            torch.save(model.state_dict(), "./ckpt/" + args.model_name + "final.pkl")

        best_metric = best_model_metrics[metric]

        if best_metric > best_config_metric:
            best_config_metric = best_metric
            best_config = dict_

        plot_config = update_tuning_records(plot_config, dict_, best_model_metrics)
        plot_config.to_csv(
            "{}_{}_tuning-report.csv".format(args.feat_extractor, args.dataset),
            index=False,
        )

        print(
            "-----------------------------------------------------------------------------------------"
        )
        print("Best Metric({}): {}".format(metric, best_metric))
        print(
            "-----------------------------------------------------------------------------------------"
        )

        if train_mode == "tuning":
            print("Curr config: {}".format(dict_))
            print("Runs: {} / {}".format(i + 1, len(param_grid)))
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

    plot_config.to_csv(
        "{}_{}_tuning-report.csv".format(args.feat_extractor, args.dataset), index=False
    )
