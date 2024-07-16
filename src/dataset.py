import numpy as np
import torch
import torch.utils.data as data

from src.utils import (
    norm_features,
    process_feat,
    sample_m_clips,
    sample_subsets_special,
)

torch.set_default_tensor_type("torch.FloatTensor")


class Dataset(data.Dataset):
    def __init__(self, args, list=None, is_normal=True, mode="test"):
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.version = args.version

        self.n_sample_clips = args.n_sample_clips
        self.L = args.L
        self.T = args.T

        self.mode = mode
        self.list = list

    def __getitem__(self, index):
        video_label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip("\n"), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if len(features.shape) < 3:
            features = np.expand_dims(features, axis=1)

        features = norm_features(features)

        if self.mode == "test":
            return features
        else:
            return self.create_snippets(features, video_label)

    def create_snippets(self, features, video_label):
        if "snippets" in self.version:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]

            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)

            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, video_label

        # csss(continuos sparse sampling strategy)
        elif self.version == "csss":
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]

            # sample L subsets with T consecutive clips
            sampled_features = sample_subsets_special(
                features, L=self.L, T=self.T
            ).mean(axis=2)

            return sampled_features, video_label

        elif self.version == "clips_multitask_sap" or self.version == "clips_multitask":
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]

            # sample m clips evenly or loop the video if is too short
            sampled_features = sample_m_clips(features, self.n_sample_clips)

            return sampled_features, video_label

    def get_label(self):
        if self.is_normal:
            video_label = torch.tensor(0.0)
        else:
            video_label = torch.tensor(1.0)

        return video_label

    def __len__(self):
        return len(self.list)
