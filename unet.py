"""U-Net model and its parts"""
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(DoubleConv, self).__init__()
        # bias = False cause we use BatchNorm
        self.double_convolution = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels,output_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_convolution(x)


class Down(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(input_channels,output_channels)
        )

    def forward(self, x):
        return self.down(x)
    

class Up(nn.Module):
    '''
    The use of Upsample or ConvTranspose2D depends on the network you are designing.
    ConvTranspose2D has trainable kernels while Upsample is a simple interpolation.
    If it is important to learn how to upscale then ConvTranspose2D should be used,
    otherwise Upsample will be enough
    '''
    def __init__(self, input_channels, output_channels, upscale = False) -> None:
        super(Up, self).__init__()
        if upscale:
            self.up = nn.ConvTranspose2d(input_channels, input_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(input_channels, output_channels)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input equals chan*height*weight
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2,diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, num_of_channels, classes=1, upscale = False) -> None:
        super(UNet, self).__init__()
        self.num_of_channels = num_of_channels
        self.num_of_classes = classes
        self.upscale = upscale

        self.start = DoubleConv(num_of_channels, 16)
        self.down1 = Down(16,32)
        self.down2 = Down(32,64)
        self.down3 = Down(64,128)
        self.down4 = Down(128,256)
        self.up1 = Up(384,128,self.upscale)
        self.up2 = Up(192,64,self.upscale)
        self.up3 = Up(96,32,self.upscale)
        self.up4 = Up(48,16,self.upscale)
        self.finish = OutConv(16, self.num_of_classes)

    def forward(self, x):
        '''
        I'm doing binary image segmentation thats why classes = 1
        and F.sigmoid is used in the end
        '''
        x1 = self.start(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return F.sigmoid(self.finish(x))


def test():
    image = torch.randn((32,3,128,128))
    model = UNet(3)
    out = model(image)
    print(image.shape, out.shape)

if __name__ == "__main__":
    test()