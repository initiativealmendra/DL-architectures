import torch
import torch.nn as nn
import torch.nn.functional as F

class mobileNet(nn.Module):
  def __init__(self):
    super().__init__()
    
    def separableConv2d(in_channels, out_channels, kernel_size):
      
      return nn.Sequential(# Depth-wise convolution
                           nn.Conv2d(in_channels,  in_channels, kernel_size, groups=in_channels),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(in_channels),
                           # Point-wise convolution
                           nn.Conv2d(in_channels, out_channels, 1),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(out_channels))
    
    
    self.conv1 = separableConv2d(3, 16, 3)  #150, 150, 3 -> 148, 148, 3 -> 148, 148, 16 
    self.conv2 = separableConv2d(16, 32, 3) 
    self.conv3 = separableConv2d(32, 64, 3)
    self.conv4 = separableConv2d(64, 128, 3)
    
    self.pool = nn.MaxPool2d(2,2)
    
    self.fc1 = nn.Linear(7*7*128, 1024)
    self.fc2 = nn.Linear(1024, 6)
    
  def forward(self, x):
    
    x = self.pool(self.conv1(x)) # 74,74,16
    x = self.pool(self.conv2(x)) # 36,36,32
    x = self.pool(self.conv3(x)) # 17,17,64
    x = self.pool(self.conv4(x)) # 7,7,128
    x = x.view(x.size()[0], -1)
    
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x))
    
    return x

n_classes = 6
batch_size = 128
img_width = 150
img_height = 150
img_depth = 3


mobilenet = mobileNet()

output = mobilenet(torch.randn([batch_size, img_depth, img_width, img_height]))
print(output.shape) # [128, 6]