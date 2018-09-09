import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self,d):
        super(Generator,self).__init__()
        self.z_size = 100
        self.conv1 = nn.ConvTranspose2d(100,d*4,4,2,0)
        self.bn1 = nn.BatchNorm2d(d*4)
        self.conv2 = nn.ConvTranspose2d(d*4,d*2,4,2,1)
        self.bn2 = nn.BatchNorm2d(d*2)
        self.conv3 = nn.ConvTranspose2d(d*2,d,4,2,1)
        self.bn3 = nn.BatchNorm2d(d)
        self.conv4 = nn.ConvTranspose2d(d,3,4,2,1)
	
 
        for m in self.modules():
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        nn.init.xavier_normal_(self.conv4.weight,nn.init.calculate_gain('tanh'))
 


    def forward(self,x):
        x = F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)),0.2)
        x = torch.tanh(self.conv4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self,d):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(3,d,4,2,1);
        self.conv2 = nn.Conv2d(d,d*2,4,2,1);
        self.conv3 = nn.Conv2d(d*2,d*4,4,2,1);
        self.conv4 = nn.Conv2d(d*4,1,4,1,0);
 
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    
    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = F.leaky_relu(self.conv3(x),0.2)
        x = F.leaky_relu(self.conv4(x),0.2)
        return x

    

