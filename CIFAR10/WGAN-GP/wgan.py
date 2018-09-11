import torch
from torch import autograd
import torchvision

class WassersteinGAN(object):
    def __init__(self,device,loader,G,D,optimG,optimD,lambd,ncritic,batch_size=1,sample_size=8,iters=10000,save_iter=10,checkpoints='model.model',recon='./images'):
        self.device = device
        self.loader = loader
        self.G = G
        self.D = D
        self.optimG = optimG
        self.optimD = optimD
        self.lmb = lambd
        self.n_critic = ncritic
        self.path = checkpoints 
        self.recon = recon 
        self.g_losses = []
        self.d_losses = []
        self.iters = iters
        self.save_iter = save_iter 
        self.start_iter = 0
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.test_noise = torch.randn(self.sample_size,self.G.z_size,1,1,device=self.device)


    def save_model(self, itr):
        print("Saving Model at '{}'".format(self.path))
        model = {
                'iter': itr+1,
                'generator': self.G.state_dict(),
                'discriminator': self.D.state_dict(),
                'g_optimizer': self.optimG.state_dict(),
                'd_optimizer': self.optimD.state_dict(),
                'g_losses': self.g_losses,
                'd_losses': self.d_losses
                }
        torch.save(model, self.path)

    def load_model(self):
        print("Loading Model From '{}'".format(self.path))
        try:
            check = torch.load(self.path)
            self.start_iter = check['iter']
            self.g_losses = check['g_losses']
            self.d_losses = check['d_losses']
            self.G.load_state_dict(check['generator'])
            self.D.load_state_dict(check['discriminator'])
            self.optimG.load_state_dict(check['g_optimizer'])
            self.optimD.load_state_dict(check['d_optimizer'])
        except:
            print("Model could not be loaded from {}. Training from Scratch".format(self.path))
            self.start_epoch= 0
            self.g_losses = []
            self.d_losses = []

    def sample_images(self,itr,nrow=8):
        with torch.no_grad():
            images = self.G(self.test_noise)
            img = torchvision.utils.make_grid(images)
            torchvision.utils.save_image(images,"%s/itr%d.png" % (self.recon,itr+1),nrow=nrow)
            
    
    def train(self):
        self.G.train()
        self.D.train()
        running_G = 0.0
        running_D = 0.0
        for itr in range(self.start_iter,self.iters):
            #Train the critic
            self.optimD.zero_grad()
            for i,data in enumerate(self.loader):
                if i >= self.n_critic:
                    break
                real,_ = data
                real = real.to(self.device)
                noise = torch.randn(self.batch_size,self.G.z_size,1,1,device=self.device)
                fake = self.G(noise)
                eps = torch.rand(1).item()
                interpolate = eps * real + (1 - eps) * fake
                
                #Calculate the critic loss D(G(z)) - D(x)
                critic_loss = self.D(fake.detach()) - - self.D(real) 
                critic_loss = critic_loss.view(self.batch_size) #The critic returns a scalar
                critic_loss = critic_loss.mean()
                critic_loss.backward()
                

                #Calculate the gradient penalty
                d_interpolate = self.D(interpolate)
                grad_outputs = torch.ones_like(d_interpolate,device=self.device)
                gradients = autograd.grad(d_interpolate,interpolate,grad_outputs=grad_outputs,create_graph=True,retain_graph=True)[0]
                gradients = gradients.view(self.batch_size,-1) #TODO:INSPECT
                grad_penalty = (gradients.norm(2,dim=1) - 1) ** 2 
                grad_penalty = grad_penalty.mean()
                grad_penalty *= self.lmb
                grad_penalty.backward()
                running_D += grad_penalty.item() + critic_loss.item()

            self.optimD.step()
            self.optimG.zero_grad()
            noise = torch.randn(self.batch_size,self.G.z_size,1,1,device=self.device)
            g_loss = self.D(self.G(noise)) * -1
            g_loss = torch.mean(g_loss)
            g_loss.backward()
            self.optimG.step()
            running_G += g_loss.item()

            if (itr+1) % self.save_iter == 0:
                self.G.eval()
                self.D.eval()
                running_G /= self.save_iter
                running_D /= self.n_critic * self.save_iter
                self.d_losses.append(running_D)
                self.g_losses.append(running_G)
                print("Iteration {} : Mean Generator Loss : {} Mean Critic Loss : {}".format(itr+1,running_G/self.save_iter,running_D/(self.save_iter*self.n_critic)))
                self.save_model(itr)
                self.sample_images(itr)
                running_G = 0.0
                running_D = 0.0
                self.G.train()
                self.D.train()
        print("Training is Complete")
