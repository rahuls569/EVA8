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
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        sc1 = self.shortcut1(x)
        x = self.convblock8(sc1)
        x = self.convblock9(x)
        x = self.convblock10(x)
        sc2 = self.shortcut2(x)
        x = self.convblock11(sc2)
        x = self.convblock12(x)
        sc3 = self.shortcut3(x)
        x = self.convblock13(sc3)
        x = self.convblock14(x)
        x = self.avgpool(x)
        x = x.view(-1, 64*6*6)
        x = self.fc(x)
        
        return x
