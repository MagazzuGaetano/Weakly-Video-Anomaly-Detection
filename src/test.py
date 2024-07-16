import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import compute_metrics


def select_vad_gt(args):
    vad_gt = None

    if args.dataset == "shanghai":
        vad_gt = np.load("list/gt/vad/gt-sh.npy")

    elif args.dataset == "ucf":
        if args.feat_extractor == "i3d":
            vad_gt = np.load("list/gt/vad/gt-ucf(i3d).npy")
        else:
            vad_gt = np.load("list/gt/vad/gt-ucf_c3d_new.npy")

    elif args.dataset == "xdv":
        if args.feat_extractor == "i3d":
            vad_gt = np.load("list/gt/vad/gt-xdv(scaricato).npy")
        else:
            vad_gt = np.load("list/gt/vad/gt-xdv.npy")

    return vad_gt


def select_vc_gt(args):
    vc_gt = None
    if args.dataset == "shanghai":
        vc_gt = np.load("list/gt/vc/train/gt-sh.npy")
    elif args.dataset == "ucf":
        vc_gt = np.load("list/gt/vc/train/gt-ucf.npy")
    elif args.dataset == "xdv":
        vc_gt = np.load("list/gt/vc/train/gt-xdv.npy")
    return vc_gt


def test(dataloader, model, args, viz, device, epoch, print_metrics=True):
    clip_rec_auc, clip_pr_auc, video_rec_auc, video_pr_auc = [None, None, None, None]

    with torch.no_grad():
        model.eval()

        clip_pred = torch.zeros(0, device=device)
        video_pred = (
            torch.zeros(0, device=device) if "multitask" in args.version else None
        )

        for i, input in enumerate(dataloader):
            input = input.to(device)  # B x T x 10 x D
            input = input.permute(0, 2, 1, 3)  # B x 10 x T x D

            if input.size()[1] == 1:
                input = input.permute(1, 0, 2, 3)  # 1 x B x T x D

            _, _, _, _, _, _, logits, _, _, _, video_score = model(inputs=input)

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits  # T x 1

            if "multitask" in args.version:
                video_score = torch.squeeze(video_score, 1)
                video_pred = torch.cat((video_pred, video_score))

            clip_pred = torch.cat((clip_pred, sig))

        # prepare weakly anomaly detection clip level prediction
        clip_pred = list(clip_pred.cpu().detach().numpy())
        clip_pred = np.repeat(np.array(clip_pred), 16)

        # select the gt for weakly anomaly detection
        vad_gt = select_vad_gt(args)

        clip_rec_auc, clip_pr_auc = compute_metrics(
            list(vad_gt), clip_pred, print_metrics=print_metrics
        )

        if epoch == -1:
            plt.figure()
            plt.plot(clip_pred)
            plt.plot(vad_gt)
            plt.show()

        # Plotting Metrics
        if print_metrics:
            viz.plot("clip_auc", "test", "Test AUC (CLIP)", epoch, clip_rec_auc)
            viz.plot("clip_pr_auc", "test", "Test PR_AUC (CLIP)", epoch, clip_pr_auc)

        if "-multitask" in args.version:
            # prepare video-level predictions
            video_pred = list(video_pred.cpu().detach().numpy())

            # select video-level labels for video classication
            video_gt_clip = select_vc_gt(args)

            video_rec_auc, video_pr_auc = compute_metrics(
                video_gt_clip, video_pred, print_metrics=print_metrics
            )

            if print_metrics:
                viz.plot("video_auc", "test", "Test AUC (VIDEO)", epoch, video_rec_auc)
                viz.plot(
                    "video_pr_auc", "test", "Test PR_AUC (VIDEO)", epoch, video_pr_auc
                )

        return clip_rec_auc, clip_pr_auc, video_rec_auc, video_pr_auc


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

    return (X.mean(), X.std())
