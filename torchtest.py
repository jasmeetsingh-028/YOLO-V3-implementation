import torch 
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act = True, **kwargs):  #if layer wants to use batch normalization and activation function
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not(bn_act), **kwargs)  # if we do not use bn and act in a layer we need bias ##kwargs - kernel size, stride
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    
    def forward(self, x):
        print("inside CNN Block")
        if self.use_bn_act:
            x = self.leaky(self.bn(self.conv(x)))
            print(" x shape after conv block: ", x.shape)
            return x
        else:
            x= self.conv(x)
            print(" x shape after conv block: ", x.shape)
            return x


if __name__ == "__main__":
    model = CNNBlock(in_channels= 512, out_channels=256, kernel_size = 1, stride = 1)
    x = torch.randn((2, 512, 13, 13))
    out = model(x)