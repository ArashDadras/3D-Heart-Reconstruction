# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Refiner(torch.nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(8192, 2048), torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192), torch.nn.ReLU()
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                128, 64, kernel_size=4, stride=2, bias=False, padding=1
            ),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                64, 32, kernel_size=4, stride=2, bias=False, padding=1
            ),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                32, 1, kernel_size=4, stride=2, bias=False, padding=1
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, coarse_volumes):
        volumes_32_l = coarse_volumes.unsqueeze(dim=1)
        volumes_16_l = self.layer1(volumes_32_l)
        volumes_8_l = self.layer2(volumes_16_l)
        volumes_4_l = self.layer3(volumes_8_l)
        flatten_features = self.layer4(volumes_4_l.view(-1, 8192))
        flatten_features = self.layer5(flatten_features)
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5

        # torch.Size([batch_size, 1, 32, 32, 32])
        return volumes_32_r.squeeze(dim=1)
