import torch

from torch.nn import MSELoss

torch.set_default_tensor_type("torch.FloatTensor")


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2 - arr) ** 2)

    return lamda1 * loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid

        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(-torch.abs(x))
        return torch.abs(
            torch.mean(-x * target + torch.clamp(x, min=0) + torch.log(tmp))
        )


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.cuda()

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        loss_abn = torch.abs(
            self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1)
        )

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total, loss_cls, loss_rtfm


class VideoClassificationLoss(torch.nn.Module):
    def __init__(self):
        super(VideoClassificationLoss, self).__init__()
        self.criterion = torch.nn.BCELoss()

    def forward(self, video_score, nlabel, alabel):
        label = torch.cat((nlabel, alabel), 0)
        label = label.cuda()

        # normal video scores + abnormal video scores
        video_score = video_score.squeeze()

        loss_cls = self.criterion(
            video_score, label
        )  # BCE loss in the video score space

        return loss_cls


def train(
    nloader,
    aloader,
    model,
    batch_size,
    optimizer,
    viz,
    device,
    epoch,
    version,
    vc_alpha,
    print_metrics=True,
):
    with torch.set_grad_enabled(True):
        model.train()

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)

        T = input.size()[2]  # 2B x 10 x T X D

        (
            score_abnormal,
            score_normal,
            feat_select_abn,
            feat_select_normal,
            feat_abn_bottom,
            feat_normal_bottom,
            scores,
            scores_nor_bottom,
            scores_nor_abn_bag,
            _,
            video_score,
        ) = model(input)  # b*T  x D

        scores = scores.view(batch_size * T * 2, -1)
        scores = scores.squeeze()

        abn_scores = scores[batch_size * T :]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_rtfm, loss1, loss2 = RTFM_loss(1e-4, model.m)(
            score_normal,
            score_abnormal,
            nlabel,
            alabel,
            feat_select_normal,
            feat_select_abn,
        )
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        loss_video = (
            VideoClassificationLoss()(video_score, nlabel, alabel)
            if "multitask" in version
            else 0
        )

        cost = loss_rtfm + loss_smooth + loss_sparse + (vc_alpha * loss_video)

        # Plotting Metrics
        if print_metrics:
            viz.plot("loss", "train", "Total Train Loss", epoch, cost.item())
            # viz.plot('loss_cls', 'train', 'Train Classification Loss', epoch, loss1.item())
            # viz.plot('loss_rtfm', 'train', 'Train RTFM Loss', epoch, loss2.item())
            # viz.plot('loss_sparsity', 'train', 'Train Loss Sparsity', epoch, loss_sparse.item())
            # viz.plot('loss_smooth', 'train', 'Train Loss Smooth', epoch, loss_smooth.item())
            if "multitask" in version:
                viz.plot(
                    "loss_vc",
                    "train",
                    "Train Video Classification Loss",
                    epoch,
                    loss_video.item(),
                )

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
