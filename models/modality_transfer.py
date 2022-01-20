import os
import torch
import torch.nn as nn
import random
import torchvision.models as models
import torch.nn.functional as F

# from model.emd_loss import EMD
class ModalityTransfer(nn.Module):
    def __init__(self, n_classes_input = 3):
        super(ModalityTransfer, self).__init__()

        # encoder
        self.conv0_1 = nn.Conv2d(3, 16, 3, stride = 1, padding = 1)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride = 1, padding = 1)
        self.conv1_1 = nn.Conv2d(16, 32, 3, stride = 2, padding = 1) # 224 -> 112
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1) # 112 -> 56
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1) # 56 -> 28
        self.conv4_1 = nn.Conv2d(128, 256, 5, stride = 2, padding = 2) # 28 -> 14
        self.conv5_1 = nn.Conv2d(256, 512, 5, stride = 2, padding = 2) # 14 -> 7

        # decoder
        self.deconv1_1 = nn.ConvTranspose2d(512,256,5,stride=2,padding=2,output_padding=1)
        self.deconv1_2 = nn.ConvTranspose2d(256,128,5,stride=2,padding=2,output_padding=1)
        self.deconv1_3 = nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1)
        self.deconv1_4 = nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1)

        # to points 
        self.conv6_1 = nn.Conv2d(64,32,3,stride=2,padding=1)
        self.conv6_2 = nn.Conv2d(32,3,3,stride=1,padding=1)

        self.bn = nn.BatchNorm2d(32)

    
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = F.relu(self.conv0_1(x))
        x = F.relu(self.conv0_2(x))
        
        x = F.relu(self.conv1_1(x))
        x1 = x # 112 
        
        x = F.relu(self.conv2_1(x))
        x2 = x # 56
        
        x = F.relu(self.conv3_1(x))
        x3 = x # 28
        
        x = F.relu(self.conv4_1(x))
        x4 = x # 14
        
        x = F.relu(self.conv5_1(x))
        x5 = x #7

        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = self.deconv1_3(x)

        x = self.conv6_1(x)
        x = self.conv6_2(x)

        points = x.view(batch_size,784,3)

        return points, [x2, x3, x4, x5] 

if __name__ == "__main__":
    net = ModalityTransfer()
    x = torch.rand([32,3,224,224])
    y,_ = net(x)
    print(y.shape)