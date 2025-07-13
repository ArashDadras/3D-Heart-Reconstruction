# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Merger(torch.nn.Module):
    def __init__(self):
        super(Merger, self).__init__()

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(36, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2),
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(0.2),
        )

    def forward(self, raw_features, coarse_volumes):
        n_views_rendering = coarse_volumes.size(1)
        raw_features = torch.split(raw_features, 1, dim=1)
        volume_weights = []

        for i in range(n_views_rendering):
            raw_feature = torch.squeeze(raw_features[i], dim=1)
            volume_weight1 = self.layer1(raw_feature)
            volume_weight2 = self.layer2(volume_weight1)
            volume_weight3 = self.layer3(volume_weight2)
            volume_weight4 = self.layer4(volume_weight3)
            volume_weight = self.layer5(
                torch.cat(
                    [
                        volume_weight1,
                        volume_weight2,
                        volume_weight3,
                        volume_weight4,
                    ],
                    dim=1,
                )
            )
            volume_weight = self.layer6(volume_weight)
            volume_weight = torch.squeeze(volume_weight, dim=1)
            volume_weights.append(volume_weight)

        volume_weights = (
            torch.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
        )
        volume_weights = torch.softmax(volume_weights, dim=1)
        # torch.Size([batch_size, n_views, 32, 32, 32])
        # torch.Size([batch_size, n_views, 32, 32, 32])
        coarse_volumes = coarse_volumes * volume_weights
        coarse_volumes = torch.sum(coarse_volumes, dim=1)

        return torch.clamp(coarse_volumes, min=0, max=1)
