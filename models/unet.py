#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 15:22
# @Author  : Eric Ching
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x

class UNet3D(nn.Module):
    """3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(UNet3D, self).__init__()
        self.squeeze_channels = 256
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()
        self.make_vae_decoder()
        self.dropout = nn.Dropout(p=p)

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.up1conv = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))

    def make_vae_decoder(self):
        init_channels = self.init_channels
        self.vconv4 = nn.Sequential(nn.GroupNorm(8, init_channels * 8),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(init_channels * 8, self.squeeze_channels, (3, 3, 3), padding=(1, 1, 1)),
                                    )

        self.mu_fc = nn.Linear(self.squeeze_channels//2, self.squeeze_channels//2)
        self.logvar_fc = nn.Linear(self.squeeze_channels//2, self.squeeze_channels//2)

        self.glob_pool = nn.AdaptiveAvgPool3d(1)
        recon_shape = np.prod(self.input_shape) // (16 ** 3)

        self.reconstraction = nn.Sequential(nn.Linear(self.squeeze_channels//2, init_channels * 8 * recon_shape),
                                            nn.ReLU(inplace=True))

        self.vup4 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 8, (1, 1, 1)),
                                  nn.Upsample(scale_factor=2))

        self.vconv3 = nn.Sequential(nn.Conv3d(init_channels * 8, init_channels * 4, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels * 4, init_channels * 4))

        self.vconv2 = nn.Sequential(nn.Conv3d(init_channels * 4, init_channels * 2, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels * 2, init_channels * 2))

        self.vconv1 = nn.Sequential(nn.Conv3d(init_channels * 2, init_channels, (3, 3, 3), padding=(1, 1, 1)),
                                    nn.Upsample(scale_factor=2),
                                    BasicBlock(init_channels, init_channels))

        self.vconv0 = nn.Conv3d(init_channels, self.in_channels, (1, 1, 1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward_vae_decoder(self, x):
        x = self.vconv4(x)
        x = self.glob_pool(x)
        batch_size = x.size()[0]
        x = x.view((batch_size, self.squeeze_channels))
        mu = x[:, :self.squeeze_channels // 2]
        mu = self.mu_fc(mu)
        logvar = x[:, self.squeeze_channels // 2:]
        logvar = self.logvar_fc(logvar)
        z = self.reparameterize(mu, logvar)
        c1 = self.reconstraction(z)
        recon_shape = [batch_size,
                       self.squeeze_channels // 2,
                       self.input_shape[0] // 16,
                       self.input_shape[1] // 16,
                       self.input_shape[2] // 16]

        c1 = c1.view(recon_shape)
        c1 = self.vup4(c1)

        x = self.vconv3(c1)
        x = self.vconv2(x)
        x = self.vconv1(x)
        vout = self.vconv0(x)

        return vout, mu, logvar

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)

        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)

        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)

        c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)

        c4d = self.dropout(c4d)

        vout, mu, logvar = self.forward_vae_decoder(c4d)

        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)
        uout = F.sigmoid(uout)

        return uout, vout, mu, logvar