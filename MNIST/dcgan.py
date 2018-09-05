import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.z_size = 100
        self.fc = nn.Linear(self.z_size,4*4*16)
        self.bn_fc = nn.BatchNorm2d(16)
        self.conv1 = nn.ConvTranspose2d(16,8,5,1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.ConvTranspose2d(8,4,5,2,2)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.ConvTranspose2d(4,1,5,2,3,1)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
               nn.init.xavier_normal_(m.weight)

        nn.init.xavier_normal_(self.conv3.weight,nn.init.calculate_gain('tanh'))
 


    def forward(self,x):
        x = self.fc(x)
        x = self.bn_fc(x.view(-1,16,4,4))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = torch.tanh(self.conv3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(1,64,5,bias=False)
        self.conv2 = nn.Conv2d(64,128,5,2,bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,128,5,2,bias=False)
        self.bn3 = nn.BatchNorm2d(128) 
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(128*9,1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
               nn.init.xavier_normal_(m.weight,nn.init.calculate_gain('sigmoid'))
    
    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)),0.2)
        x = x.view(-1,9*128)
        x = F.sigmoid(self.fc(self.drop(x)))
        return x

    

