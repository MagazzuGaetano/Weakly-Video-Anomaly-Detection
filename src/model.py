import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

torch.set_default_tensor_type("torch.FloatTensor")


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class _NonLocalBlockND(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        dimension=3,
        sub_sample=True,
        bn_layer=True,
    ):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(
        self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True
    ):
        super(NONLocalBlock1D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=1,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=int(len_feature / 4),
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
            ),
            nn.ReLU(),
            bn(int(len_feature / 4)),
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=int(len_feature / 4),
                kernel_size=3,
                stride=1,
                dilation=2,
                padding=2,
            ),
            nn.ReLU(),
            bn(int(len_feature / 4)),
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=int(len_feature / 4),
                kernel_size=3,
                stride=1,
                dilation=4,
                padding=4,
            ),
            nn.ReLU(),
            bn(int(len_feature / 4)),
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=int(len_feature / 4),
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # bias=False
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(
                in_channels=len_feature,
                out_channels=len_feature,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # should we keep the bias? # bias=False
            nn.ReLU(),
            bn(len_feature),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(
            int(len_feature / 4), sub_sample=False, bn_layer=True
        )

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)
        out3 = self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim=1)
        out = self.conv_4(out)
        out = self.non_local(out)
        out = torch.cat((out_d, out), dim=1)
        out = self.conv_5(out)  # fuse all the features together
        out = out + residual
        out = out.permute(0, 2, 1)
        # out: (B, T, 1)

        return out


class SelfAttentivePooling(nn.Module):
    def __init__(self, input_dim, embed_dim, attn_dropout=0.7):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.drop_out = nn.Dropout(attn_dropout)
        self.W = nn.Linear(input_dim, embed_dim)
        self.u = torch.nn.Parameter(torch.randn(embed_dim))

    def forward(self, inputs):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, H')

        return:
            utter_rep: size (N, H)
        """
        lin_out = self.W(inputs)
        v_view = self.u.unsqueeze(0).expand(lin_out.size(0), len(self.u)).unsqueeze(2)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze())
        attention_weights = self.drop_out(attention_weights)

        if inputs.shape[1] == 1:
            attention_weights = attention_weights.unsqueeze(1)

        attention_weights_normalized = F.softmax(attention_weights, 1)
        utter_rep = torch.sum(inputs * attention_weights_normalized.unsqueeze(2), dim=1)
        return utter_rep


class Model(nn.Module):
    def __init__(
        self, n_features, batch_size, k, m, sap_hidden_size, sap=False, multitask=False
    ):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.k_abn = k
        self.k_nor = k
        self.m = m
        self.sap = sap
        self.multitask = multitask
        self.flatten = nn.Flatten()

        self.Aggregate = Aggregate(len_feature=n_features)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.SAP = SelfAttentivePooling(input_dim=n_features, embed_dim=sap_hidden_size)
        self.fc4 = nn.Linear(n_features, 512)
        self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.params = nn.ModuleDict(
            {
                "temporal_features": nn.ModuleList([self.Aggregate]),
                "snippet_classifier": nn.ModuleList([self.fc1, self.fc2, self.fc3]),
                "video_classifier": nn.ModuleList(
                    [self.SAP, self.fc4, self.fc5, self.fc6]
                ),
            }
        )

        self.apply(weight_init)

    def forward(self, inputs):
        k_abn = self.k_abn
        k_nor = self.k_nor

        # [32, 10, 32, 3, 1024]
        out = inputs

        bs, ncrops, t, f = out.size()  # [32, 10, 32, 1024]
        out = out.view(-1, t, f)  # [320, 32, 1024]

        # Multi-Scale Temporal Network
        out = self.Aggregate(out)
        out = self.drop_out(out)
        features = out  # [320, 32, 1024]

        if self.multitask:
            # aggregate the temporal dimension with a sap layer or gap layer
            if self.sap:
                aggregated_features = self.SAP(features)  # [320, 1024]
            else:
                aggregated_features = features.mean(1)  # [320, 1024]

            aggregated_features = self.relu(self.fc4(aggregated_features))
            aggregated_features = self.drop_out(aggregated_features)
            aggregated_features = self.relu(self.fc5(aggregated_features))
            aggregated_features = self.drop_out(aggregated_features)

            # video classification scores for normal videos + abnormal videos
            video_score = self.sigmoid(self.fc6(aggregated_features))  # [320, 1]

            # average score for 10 crops
            video_score = video_score.view(bs, ncrops, -1).mean(1)  # [32, 1]
        else:
            video_score = None

        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))  # [320, 32, 1]

        scores = scores.view(bs, ncrops, -1).mean(1)
        scores = scores.unsqueeze(dim=2)  # [32, 32, 1]

        if self.multitask:
            z = video_score.unsqueeze(1).expand(
                (video_score.size(0), scores.size(1), 1)
            )
            scores = scores * z

        score_abnormal = None
        score_normal = None
        feat_select_abn = None
        feat_select_normal = None
        feat_select_abn = None
        feat_select_abn = None
        feat_select_abn = None
        feat_select_abn = None
        feat_magnitudes = None

        if self.training:
            normal_features = features[0 : self.batch_size * ncrops]  # [320, 32, 1024]
            normal_scores = scores[0 : self.batch_size]  # [320, 32, 1]

            abnormal_features = features[self.batch_size * ncrops :]  # [320, 32, 1024]
            abnormal_scores = scores[self.batch_size :]  # [320, 32, 1]

            feat_magnitudes = torch.norm(features, p=2, dim=2)  # [320, 32, 1]
            feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [32, 32]

            # normal feature magnitudes
            nfea_magnitudes = feat_magnitudes[0 : self.batch_size]

            # abnormal feature magnitudes
            afea_magnitudes = feat_magnitudes[self.batch_size :]

            n_size = nfea_magnitudes.shape[0]

            # this is for inference, the batch size is 1
            if nfea_magnitudes.shape[0] == 1:
                afea_magnitudes = nfea_magnitudes
                abnormal_scores = normal_scores
                abnormal_features = normal_features

            select_idx = torch.ones_like(nfea_magnitudes)
            select_idx = self.drop_out(select_idx)

            #######  process abnormal videos -> select top3 feature magnitude  #######
            afea_magnitudes_drop = afea_magnitudes * select_idx
            idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
            idx_abn_feat = idx_abn.unsqueeze(2).expand(
                [-1, -1, abnormal_features.shape[2]]
            )

            abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
            abnormal_features = abnormal_features.permute(1, 0, 2, 3)

            # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.zeros(0, device=inputs.device)
            for abnormal_feature in abnormal_features:
                feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)
                total_select_abn_feature = torch.cat(
                    (total_select_abn_feature, feat_select_abn)
                )

            # top 3 scores in abnormal bag based on the top-3 magnitude
            idx_abn_score = idx_abn.unsqueeze(2).expand(
                [-1, -1, abnormal_scores.shape[2]]
            )
            score_abnormal = torch.mean(
                torch.gather(abnormal_scores, 1, idx_abn_score), dim=1
            )

            ####### process normal videos -> select top3 feature magnitude #######
            select_idx_normal = torch.ones_like(nfea_magnitudes)
            select_idx_normal = self.drop_out(select_idx_normal)
            nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
            idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
            idx_normal_feat = idx_normal.unsqueeze(2).expand(
                [-1, -1, normal_features.shape[2]]
            )

            normal_features = normal_features.view(n_size, ncrops, t, f)
            normal_features = normal_features.permute(1, 0, 2, 3)

            # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.zeros(0, device=inputs.device)
            for nor_fea in normal_features:
                feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)
                total_select_nor_feature = torch.cat(
                    (total_select_nor_feature, feat_select_normal)
                )

            # top 3 scores in normal bag
            idx_normal_score = idx_normal.unsqueeze(2).expand(
                [-1, -1, normal_scores.shape[2]]
            )
            score_normal = torch.mean(
                torch.gather(normal_scores, 1, idx_normal_score), dim=1
            )

            feat_select_abn = total_select_abn_feature
            feat_select_normal = total_select_nor_feature

        return (
            score_abnormal,
            score_normal,
            feat_select_abn,
            feat_select_normal,
            feat_select_abn,
            feat_select_abn,
            scores,
            feat_select_abn,
            feat_select_abn,
            feat_magnitudes,
            video_score,
        )
