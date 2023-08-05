import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import numpy as np
import gc
import pandas as pd


def compute_metrics(gt, pred):

    fpr, tpr, th = roc_curve(gt, pred)
    rec_auc = auc(fpr, tpr)
    print("auc : " + str(rec_auc))

    # precision, recall, th = precision_recall_curve(gt, pred)
    # pr_auc = auc(recall, precision)
    # print('pr_auc : ' + str(pr_auc))

    # print('precision : ' + str(np.asarray(precision).mean()))
    # print('recall : ' + str(np.asarray(recall).mean()))
    # print('pr_auc : ' + str(pr_auc))

    # 'micro', 'samples', 'weighted', 'macro'
    ap = average_precision_score(gt, pred, pos_label=1)
    print("ap : " + str(ap))

    return rec_auc, ap


def test(dataloader, model, args, viz, device, epoch, print_metrics=True):
    with torch.no_grad():
        model.eval()
        clip_pred = torch.zeros(0, device=device)
        video_pred = (
            torch.zeros(0, device=device) if args.version == "sample_SAP" else None
        )

        for i, input in enumerate(dataloader):
            input = input.to(device)  # B x T x 10 x D
            input = input.permute(0, 2, 1, 3)  # B x 10 x T x D

            if input.size()[1] == 1:
                input = input.permute(1, 0, 2, 3)  # 1 x B x T x D

            _, _, _, _, _, _, logits, _, _, _, video_score = model(inputs=input)

            if args.version == "sample_SAP":
                video_score = torch.squeeze(video_score, 1)
                video_pred = torch.cat((video_pred, video_score))

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits  # T x 1
            clip_pred = torch.cat((clip_pred, sig))

        if args.dataset == "shanghai":
            vad_gt = np.load("list/gt/vad/gt-sh.npy")
            vc_gt = np.load("list/gt/vc/gt-sh.npy")

        elif args.dataset == "ucf":
            if args.feat_extractor == "i3d":
                vad_gt = np.load("list/gt/vad/gt-ucf(i3d).npy")
            else:
                vad_gt = np.load("list/gt/vad/gt-ucf_c3d_new.npy")

            vc_gt = np.load("list/gt/vc/gt-ucf.npy")

        elif args.dataset == "xdv":
            if args.feat_extractor == "i3d":
                vad_gt = np.load("list/gt/vad/gt-xdv(scaricato).npy")
            else:
                vad_gt = np.load("list/gt/vad/gt-xdv.npy")

            vc_gt = np.load("list/gt/vc/gt-xdv.npy")

        clip_pred = list(clip_pred.cpu().detach().numpy())
        clip_pred = np.repeat(np.array(clip_pred), 16)
        clip_rec_auc, clip_pr_auc = compute_metrics(list(vad_gt), clip_pred)

        # to plot predicted scores
        # if epoch == 1000:
        #     plt.figure()
        #     plt.plot(clip_pred)
        #     plt.plot(vad_gt)
        #     plt.show()

        # Plotting Metrics
        if print_metrics:
            viz.plot("clip_auc", "test", "Test AUC (CLIP)", epoch, clip_rec_auc)
            viz.plot("clip_pr_auc", "test", "Test PR_AUC (CLIP)", epoch, clip_pr_auc)

        if args.version == "sample_SAP":
            video_pred = list(video_pred.cpu().detach().numpy())
            video_rec_auc, video_pr_auc = compute_metrics(list(vc_gt), video_pred)

            if print_metrics:
                viz.plot("video_auc", "test", "Test AUC (VIDEO)", epoch, video_rec_auc)
                viz.plot(
                    "video_pr_auc", "test", "Test PR_AUC (VIDEO)", epoch, video_pr_auc
                )

        return clip_rec_auc, clip_pr_auc
