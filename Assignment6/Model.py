import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)        # input 32X32X3 || 3X3X3X16 || 16X16X16       RF= 1+(3-1)*2=5                           #output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)       #  16X16X16 ||  3X3X16X32  || 8X8X32          RF= 5+(3-1)*4=13
        self.depthwise = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)   #8X8X32 || 3X3X32X32 || 4X4X32       RF= 13+(3-1)*8=29   
        self.pointwise = nn.Conv2d(32, 64, kernel_size=1)            # 4X4X32 || 1X1X32X64 || 4X4X64                     RF= 29
        self.conv3 = nn.Conv2d(64, 128, 3, dilation=2, padding=2)    # 4X4X64 ||                                           RF= 29+(3-1)*16=29+32=61
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.depthwise(x))
        x = F.relu(self.pointwise(x))
        x = F.relu(self.conv3(x))
        x = self.avg_pool(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
