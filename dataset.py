import torch.utils.data as data
import numpy as np
import torch
import os

from utils import (
    norm_features,
    process_feat,
    sample_m_clips,
    sample_subsets_special,
)
from torch.utils.data import DataLoader

torch.set_default_tensor_type("torch.FloatTensor")


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.version = args.version
        self.n_sample_clips = args.n_sample_clips
        self.L = args.L
        self.T = args.T

        if self.dataset == "shanghai":
            if test_mode:
                self.rgb_list_file = "list/shanghai-i3d-test-10crop.list"
            else:
                self.rgb_list_file = "list/shanghai-i3d-train-10crop.list"

        elif self.dataset == "ucf":
            if test_mode:
                self.rgb_list_file = "list/ucf-i3d-test.list"
            else:
                self.rgb_list_file = "list/ucf-i3d.list"

        elif self.dataset == "xdv":
            if test_mode:
                self.rgb_list_file = "list/xdv_rgb_test.list"
            else:
                self.rgb_list_file = "list/xdv_rgb.list"

        self.tranform = transform
        self.test_mode = test_mode
        self.list = self._parse_list()

        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        out_list = []

        with open(self.rgb_list_file) as f:
            rgb_list_file = f.readlines()
            out_list = rgb_list_file

        if self.test_mode is False:
            if self.dataset == "shanghai":
                if self.is_normal:
                    out_list = rgb_list_file[63:]
                    print(len(out_list))
                    print("normal list for shanghai tech")
                else:
                    out_list = rgb_list_file[:63]
                    print(len(out_list))
                    print("abnormal list for shanghai tech")

            elif self.dataset == "ucf":
                if self.is_normal:
                    out_list = rgb_list_file[810:]
                    print("normal list for ucf")
                else:
                    out_list = rgb_list_file[:810]
                    print("abnormal list for ucf")

            elif self.dataset == "xdv":
                if self.is_normal:
                    out_list = [f for f in rgb_list_file if "_label_A" in f]
                    out_list = [
                        f for f in out_list if "v=8cTqh9tMz_I__#1_label_A" not in f
                    ]  # discard corrupted files
                    print("normal list for xdv")
                else:
                    out_list = [f for f in rgb_list_file if "_label_A" not in f]
                    print("abnormal list for xdv")

        return out_list

    def __getitem__(self, index):
        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip("\n"), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if len(features.shape) < 3:
            features = np.expand_dims(features, axis=1)

        features = norm_features(features)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            return self.create_snippets(features, label)

    def create_snippets(self, features, label):
        if self.version == "snippets":
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]

            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)

            divided_features = np.array(divided_features, dtype=np.float32)
            return divided_features, label

        elif self.version == "CSSS":
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]

            # sample L subsets with T consecutive clips
            sampled_features = sample_subsets_special(
                features, L=self.L, T=self.T
            ).mean(axis=2)
            return sampled_features, label

        elif self.version == "sample_SAP":
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]

            # sample m clips evenly or loop the video if is too short
            sampled_features = sample_m_clips(features, self.n_sample_clips)
            return sampled_features, label

    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
