import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.activation = nn.ReLU() if activation else nn.Identity()
        
    def forward(self, x):
        return self.activation(self.conv(x))
# ---------------- ResBlock ----------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels


        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same")
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same")
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu2 = nn.ReLU()

        # projection nếu số kênh thay đổi
        if self.in_channels != self.out_channels:
            self.proj = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.proj = None

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.relu2(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.conv = ConvBlock(in_channels, in_channels, dilation)
        
    def forward(self, x, skip):
        return self.conv(x + skip)
# ---------------- DecoderBlock ----------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.resblocks = nn.Sequential(
            ResBlock(in_channels, in_channels), 
            ResBlock(in_channels, out_channels)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        x = self.resblocks(x)
        if skip is not None:
            x = x + skip  # Skip connection
        return x

class Resnet34(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet34, self).__init__()
        # -------- Encoder ----------
        self.conv0 = nn.Conv1d(1, 48, 80, 4)
        self.bn0 = nn.BatchNorm1d(48)
        self.relu = nn.ReLU()
        self.pool0 = nn.MaxPool1d(4)

        self.stage0 = nn.Sequential(ResBlock(48, 48), ResBlock(48, 48))
        self.pool1 = nn.MaxPool1d(4)
        self.stage1 = nn.Sequential(ResBlock(48, 96), ResBlock(96, 96))
        self.pool2 = nn.MaxPool1d(4)
        self.stage2 = nn.Sequential(ResBlock(96, 192), ResBlock(192, 192))
        self.pool3 = nn.MaxPool1d(4)
        self.stage3 = nn.Sequential(ResBlock(192, 384), ResBlock(384, 384))
        self.avgpool = nn.AvgPool1d(1)
        

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out0 = self.pool0(out)   #48, 620

        out1 = self.stage0(out0)
        out1p = self.pool1(out1) #48, 155

        out2 = self.stage1(out1p)
        out2p = self.pool2(out2) #96, 39

        out3 = self.stage2(out2p)
        out3p = self.pool3(out3) #192, 9

        out4 = self.stage3(out3p) #384, 9
        out4 = self.avgpool(out4) 
        return out3p, out2p, out1p, out0, out4
# ---------------- ResNet34 Encoder-Decoder ----------------
class ResNet34_Autoencoder(nn.Module):
    def __init__(self, num_classes=2, training = False):
        super(ResNet34_Autoencoder, self).__init__()
        self.training = training
        # -------- Encoder ----------
        self.encoder = Resnet34(num_classes)
        # ######## Where we produce the mu and log_var 
        self.conv_mu = nn.Conv1d(
            in_channels=384, 
            out_channels=384, 
            kernel_size=1
        )
        self.conv_log_var = nn.Conv1d(
            in_channels=384, 
            out_channels=384, 
            kernel_size=1
        )
        # self.conv_mu  = nn.Conv1d(384, 96, 1, 1)
        # self.conv_log_var = nn.Conv1d(384, 96, 1, 1)
        self.act_fnc = nn.ELU()

        # -------- Decoder ----------
        self.dec3 = DecoderBlock(384, 192, kernel_size=3, stride=1, padding=1, output_padding=0)   # đối xứng stage3 -> stage2
        self.dec2 = DecoderBlock(192, 96,  kernel_size=3, stride=4, padding=0,output_padding=3)    # đối xứng stage2 -> stage1
        self.dec1 = DecoderBlock(96, 48, kernel_size=4, stride=4, padding=0, output_padding=3)     # đối xứng stage1 -> stage0
        self.dec0 = DecoderBlock(48, 48, kernel_size=4, stride=4, padding=0, output_padding=0)     # đối xứng stage0 -> input

        # Reconstruction
        self.recon = nn.ConvTranspose1d(48, 1, kernel_size=96, stride=16, padding=0)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def forward(self, x):
        # ------ Encoder ------
        out3p, out2p, out1p, out0, out4 = self.encoder(x)
        # print("OUT4: ", out4.shape)
        mu = self.conv_mu(out4)
        log_var = self.conv_log_var(out4)
        # print("mu: ", mu.shape) #torch.Size([32, 384, 9])
        # if self.training:
        out4 = self.sample(mu, log_var)
        # print("out4: ", out4.shape)
        # else:
        #     out4 = mu
        # print("OKAY")
        # ------ Decoder ------
        d3 = self.dec3(out4, skip=out3p)
        d2 = self.dec2(d3, skip=out2p)
        d1 = self.dec1(d2, skip=out1p)
        d0 = self.dec0(d1, skip=out0)

        recon = self.recon(d0)

        return recon, mu, log_var

