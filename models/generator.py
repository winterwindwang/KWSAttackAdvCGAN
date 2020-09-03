import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=31, stride=1, padding=15)
        self.in1 = nn.InstanceNorm1d(channels, affine=True)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=31, stride=1, padding=15)
        self.in2 = nn.InstanceNorm1d(channels, affine=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))

        out = out + residual

        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)

        padding = kernel_size // 2

        self.conv2d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):

        if self.upsample:
            x = self.upsample_layer(x)

        x = self.conv2d(x)

        return x


class CGenerator(nn.Module):
    def __init__(self):
        super(CGenerator, self).__init__()
        # self.label_emb = nn.Embedding(num_embeddings=10, embedding_dim=10)
        self.embedding1 = nn.Embedding(10, 10)
        self.embedding2 = nn.Linear(in_features=10, out_features=16384)
        self.enc1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=32, stride=2,
                              padding=15)  # 8192
        self.enc1_nl = nn.ReLU() #  nn.BatchNorm1d(16) # nn.Tanh() # nn.PReLU()
        # (in_channels,out_channels, kernel_size,stride, padding)
        self.enc2 = nn.Conv1d(16, 32, 32, 2, 15)  # 4096
        self.enc2_nl = nn.ReLU()  # nn.BatchNorm1d(32) #  nn.Tanh()   #

        self.enc3 = nn.Conv1d(32, 32, 32, 2, 15)  # 2048
        self.enc3_nl = nn.ReLU()  # nn.BatchNorm1d(32) #nn.Tanh() #

        self.enc4 = nn.Conv1d(32, 64, 32, 2, 15)  # 1024
        self.enc4_nl = nn.ReLU()  # nn.BatchNorm1d(64) #nn.Tanh() #
        self.enc5 = nn.Conv1d(64, 64, 32, 2, 15)  # 512
        self.enc5_nl = nn.ReLU()  # nn.BatchNorm1d(64) # nn.Tanh() #
        self.enc6 = nn.Conv1d(64, 128, 32, 2, 15)  # 256
        self.enc6_nl = nn.ReLU()  # nn.BatchNorm1d(128) #nn.Tanh() #
        self.enc7 = nn.Conv1d(128, 128, 32, 2, 15)  # 128
        self.enc7_nl = nn.ReLU() #  nn.BatchNorm1d(128) #nn.Tanh()#
        self.enc8 = nn.Conv1d(128, 256, 32, 2, 15)  # 64
        self.enc8_nl = nn.ReLU()  # nn.BatchNorm1d(256) #nn.Tanh() #
        # (in_channels, out_channels, kernel_size, stride, padding)
        self.dec7 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # 128
        self.dec7_nl = nn.ReLU()  #  nn.BatchNorm1d(256) # nn.Tanh() #
        self.dec6 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # 256
        self.dec6_nl = nn.ReLU()  # nn.BatchNorm1d(256) #nn.Tanh()#
        self.dec5 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  # 512
        self.dec5_nl = nn.ReLU()  # nn.BatchNorm1d(128) # nn.Tanh() #
        self.dec4 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # 1024
        self.dec4_nl = nn.ReLU() # nn.BatchNorm1d(128) #nn.Tanh() #
        self.dec3 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # 2048
        self.dec3_nl = nn.ReLU()  # nn.BatchNorm1d(64) # nn.Tanh() #
        self.dec2 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  # 4096
        self.dec2_nl = nn.ReLU()   #  nn.BatchNorm1d(64) # nn.Tanh() #
        self.dec1 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # 8192
        self.dec1_nl = nn.ReLU()  # nn.BatchNorm1d(32) # nn.Tanh() #
        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # 16384
        self.dec_tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)
        # nn.init.normal_(self.embedding1.weight.data, 0, 0.01)
        # nn.init.normal_(self.embedding2.weight.data, 0, 0.01)


    def forward(self, x, label):
        laten = self.embedding2(self.embedding1(label))
        x = torch.cat((x, torch.unsqueeze(laten, dim=1)),dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        d7 = self.dec7(e8)
        d6 = self.dec6(torch.cat((d7, e7), dim=1))
        d5 = self.dec5(torch.cat((d6, e6), dim=1))
        d4 = self.dec4(torch.cat((d5, e5), dim=1))
        d3 = self.dec3(torch.cat((d4, e4), dim=1))
        d2 = self.dec2(torch.cat((d3, e3), dim=1))
        d1 = self.dec1(torch.cat((d2, e2), dim=1))
        out = self.dec_tanh(self.dec_final(torch.cat((d1, e1), dim=1)))
        return out
