import torch
import torch.nn as nn
from torchsummary import summary

# config (out_ch, in_ch, stride)
""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1), #(out_ch, kernel size, stride)
    (64, 3, 2),
    ["B", 1],   #["residual block", "no of repeats"]
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  #To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",       #scale prediction block   (13*13 grid)
    (256, 1, 1),
    "U",       #upsampling
    (256, 1, 1),
    (512, 3, 1),
    "S",       #scale prediction block   (26*26 grid)
    (128, 1, 1),
    "U",       #upsampling
    (128, 1, 1),
    (256, 3, 1),
    "S",       #scale prediction block   (52*52 grid)
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act = True, **kwargs):  #if layer wants to use batch normalization and activation function
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not(bn_act), **kwargs)  # if we do not use bn and act in a layer we need bias ##kwargs - kernel size, stride
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    
    def forward(self, x):
        #print("inside CNN Block")
        if self.use_bn_act:
            x = self.leaky(self.bn(self.conv(x)))
            #print(" x shape after conv block: ", x.shape)
            return x
        else:
            x= self.conv(x)
            #print(" x shape after conv block: ", x.shape)
            return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual = True, num_repeats = 1):
        super().__init__()
        self.Layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.Layers += [
                nn.Sequential(
                CNNBlock(channels, channels//2, kernel_size = 1),
                CNNBlock(channels//2, channels, kernel_size = 3, padding =1)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        #print("inside Residual block")
        for layer in self.Layers:
            if self.use_residual:
                x = layer(x) + x          
            else:
                x = layer(x)

        #print(" x shape after Residual block: ", x.shape)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2* in_channels, kernel_size = 3, padding = 1),
            CNNBlock(2*in_channels, (num_classes + 5) * 3, bn_act = False, kernel_size = 1)     #for 3 anchor boxes * [pc, x, y, w, h]
        )
        self.num_classes = num_classes

    def forward(self, x):
        #print("Inside scale prediction")
        x = self.pred(x).reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3]).permute(0,1,3,4,2)     #x.shape[0] - batch size, 3 # outputs for different anchor boxes, num_classes+5 #.permute for reordering channels
        #print(" x shape after Scale prediction block: ", x.shape)
        return(x)
    
    #N - examples, 3 - anchor boxes, 13*13 grid, 5+num_classes output

class YOLOv3(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 20 ): #pascal voc 20 classes
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self.create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):  #storing output after scale prediction
                outputs.append(layer(x))
                #print()
                #print("__"*40)
                continue    #this is the game  

            x = layer(x)  #x is not stored after scale prediction block x as the dim of the layer previous to scale prediction which is the output of the cnn block
            

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                #print("UPSAMPLED BY 2")
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
            #print("__"*40)

        return outputs
 
    def create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:  #checking for list tuple or string
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                    in_channels,
                    out_channels, 
                    kernel_size = kernel_size, 
                    stride = stride, 
                    padding = 1 if kernel_size ==3 else 0,

                    )
                )
                in_channels = out_channels
            
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                    in_channels, 
                    num_repeats=num_repeats))
            
            elif isinstance(module, str):
                if module == 'S':
                    layers += [
                        ResidualBlock(in_channels, use_residual= False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size = 1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
            
        return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")