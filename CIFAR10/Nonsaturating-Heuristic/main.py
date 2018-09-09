from dcgan import *
from nonsaturating import *
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
mnist = torchvision.datasets.MNIST('./MNIST',transform=transforms.Compose([transforms.Scale(32),transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))]),download=True)
generator = Generator(64)
discriminator = Discriminator(64)
device = torch.device('cuda:0')
generator.to(device)
discriminator.to(device)
print(len(mnist))
loader = torch.utils.data.DataLoader(mnist,batch_size=32,num_workers=4)
optimG = optim.Adam(generator.parameters(),lr=0.0002)
optimD = optim.SGD(discriminator.parameters(),lr=0.01)
gan = NonsaturatingGAN(device,loader,generator,discriminator,optimG,optimD,batch_size=32,sample_size=32,epochs=100,label_smooth=0.7,p_flip=0.05,checkpoints='./nonsaturating-gan.model',recon='./images/')
gan.set_targets(1)
gan.load_model()
gan.train()
