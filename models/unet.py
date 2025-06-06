import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .custom_conv import CustomConv2d
    _HAS_CUSTOM = True
except Exception:
    CustomConv2d = nn.Conv2d
    _HAS_CUSTOM = False

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        Conv = CustomConv2d if _HAS_CUSTOM else nn.Conv2d
        self.double_conv = nn.Sequential(
            Conv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            Conv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024 // factor)
        )
        self.up1 = nn.ConvTranspose2d(1024 // factor, 512 // factor, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512 // factor)
        self.up2 = nn.ConvTranspose2d(512 // factor, 256 // factor, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256 // factor)
        self.up3 = nn.ConvTranspose2d(256 // factor, 128 // factor, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128 // factor)
        self.up4 = nn.ConvTranspose2d(128 // factor, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv4(x)
        
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up_conv1 = torch.utils.checkpoint.checkpoint(self.up_conv1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up_conv2 = torch.utils.checkpoint.checkpoint(self.up_conv2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up_conv3 = torch.utils.checkpoint.checkpoint(self.up_conv3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.up_conv4 = torch.utils.checkpoint.checkpoint(self.up_conv4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc) 