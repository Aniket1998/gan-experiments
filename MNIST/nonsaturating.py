import gan
import torch
import torch.nn.functional as F


class NonsaturatingGAN(gan.GAN):
    def set_targets(self,target_dim):
        self.target_g = torch.ones(self.batch_size,target_dim,1,1,device=self.device)
        self.target_real = self.label_smooth * torch.ones(self.batch_size,target_dim,1,1,device=self.device)
        self.target_fake = torch.zeros(self.batch_size,target_dim,1,1,device=self.device)
        self.test_noise = torch.randn(self.sample_size,target_dim,1,1,device=self.device)

    def train_loop(self,images,labels):
        #Train the discriminator on the real examples
        self.optimD.zero_grad()
        d_real = self.D(images)
        if torch.rand(1).item() >= self.p_flip:
            loss_real = F.binary_cross_entropy(d_real,self.target_real) 
        else:
            loss_real = F.binary_cross_entropy(d_real,self.target_fake)
        loss_real.backward()
        
        #Train the discriminator on the fake examples
        noise = torch.randn(self.batch_size,self.G.z_size,1,1,device=self.device)
        fake = self.G(noise)
        d_fake = self.D(fake.detach())
        if torch.rand(1).item() >= self.p_flip:
            loss_fake = F.binary_cross_entropy(d_fake,self.target_fake)
        else:
            loss_fake = F.binary_cross_entropy(d_fake,self.target_real)
        loss_fake.backward()
        self.optimD.step()

        #Train the generator
        self.optimG.zero_grad()
        noise = torch.randn(self.batch_size,self.G.z_size,1,1,device=self.device)
        g_fake = self.D(self.G(noise))
        loss_g = F.binary_cross_entropy(g_fake,self.target_g)
        loss_g.backward()
        self.optimG.step()
        return loss_real.item()+loss_fake.item(),loss_g.item()

        

        
