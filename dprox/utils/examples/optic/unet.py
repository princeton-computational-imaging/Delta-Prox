import torch.nn as nn
import torch.utils.data
import torch


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.conv(x) + self.conv_residual(x)
        return x


class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(U_Net, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Down1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Down2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Down3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Down4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out+x
