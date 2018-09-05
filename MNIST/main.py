from dcgan import *
from nonsaturating import *
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
mnist = torchvision.datasets.MNIST('./MNIST',transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))]),download=True)
generator = Generator()
discriminator = Discriminator()
device = torch.device('cuda:0')
generator.to(device)
discriminator.to(device)
print(len(mnist))
loader = torch.utils.data.DataLoader(mnist,batch_size=32,num_workers=4)
optimG = optim.Adam(generator.parameters(),lr=0.0002)
optimD = optim.Adam(discriminator.parameters(),lr=0.0002)
gan = NonsaturatingGAN(device,loader,generator,discriminator,optimG,optimD,batch_size=32,epochs=40,label_smooth=0.8,checkpoints='./nonsaturating-gan.model',recon='./images/')
gan.set_targets(1)
gan.load_model()
gan.train()
