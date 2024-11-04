%%writefile unet.py

import torch
import torch.nn as nn


class UNetEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNetEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        pooled = self.pool(x)
        return x, pooled


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UNetOutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetOutputBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, input_channels=3, output_channels=1):
        super(UNet, self).__init__()
        self.encoder1 = UNetEncoderBlock(input_channels, 64)
        self.encoder2 = UNetEncoderBlock(64, 128)
        self.encoder3 = UNetEncoderBlock(128, 256)
        self.encoder4 = UNetEncoderBlock(256, 512)

        self.bottleneck = UNetEncoderBlock(512, 1024)

        self.decoder4 = UNetDecoderBlock(1024, 512)
        self.decoder3 = UNetDecoderBlock(512, 256)
        self.decoder2 = UNetDecoderBlock(256, 128)
        self.decoder1 = UNetDecoderBlock(128, 64)

        self.output = UNetOutputBlock(64, output_channels)

    def forward(self, x):
        enc1, pool1 = self.encoder1(x)
        enc2, pool2 = self.encoder2(x)
        enc3, pool3 = self.encoder3(x)
        enc4, pool4 = self.encoder4(x)

        bottleneck, _ = self.bottleneck(pool4)

        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        return self.output(dec1)