from dcgan import *
from wgan import *
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
cifar = torchvision.datasets.CIFAR10('./CIFAR',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))]),download=True)
generator = Generator(128)
discriminator = Discriminator(128)
device = torch.device('cuda:0')
generator.to(device)
discriminator.to(device)
print(len(cifar))
optim.Adam
loader = torch.utils.data.DataLoader(cifar,batch_size=64,shuffle=True,num_workers=4)
optimG = optim.Adam(generator.parameters(),lr=0.0001,betas=(0.0,0.9))
optimD = optim.Adam(discriminator.parameters(),lr=0.0001,betas=(0.0,0.9))
wgan = WassersteinGAN(device,loader,generator,discriminator,optimG,optimD,10.0,5,64,64,200000,100,'./WGAN-GP.model')
#gan.load_model()
wgan.train()
