import torch.nn.functional as F
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        ) 

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
        ) 

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
        ) 

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
           
        ) 

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
        ) 
            
        # CONVOLUTION BLOCK 3       
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
        ) 

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
        )

        # TRANSITION BLOCK 3
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, dilation=2, padding=0),
            nn.BatchNorm2d(64),
        ) 
        
        # CONVOLUTION BLOCK 4       
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=6*64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(6*64),
            nn.Conv2d(in_channels=6*64, out_channels=6*64, kernel_size=3, stride=1, padding=1, groups=6*64),
            nn.ReLU(),
            nn.BatchNorm2d(6*64),
            nn.Conv2d(in_channels=6*64, out_channels=132, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(132),
           
        ) 

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 132, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(132),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) #o/p size = 512*1*1 RF = 92

        self.linear = nn.Linear(132, 10)


    def forward(self, x):
        x1 = self.convblock1(x)

        x2 = self.convblock2(x1)
        x3 = x2 + x1

        x4 = self.convblock3(x3)

        x5 = self.convblock4(x4)
        x6 = x5 + x4

        x7 = self.convblock6(x6)

        x8 = self.convblock7(x7)
        x9 = x8 + self.shortcut1(x7)

        x10 = self.convblock8(x9)

        x11 = self.convblock9(x10)
        x12 = x11 + self.shortcut2(x10)


        out = self.gap(x12)        
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
