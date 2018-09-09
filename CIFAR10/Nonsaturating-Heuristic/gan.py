import torch
import torchvision

class GAN(object):
    def __init__(self,device,loader,G,D,optimG,optimD,batch_size=1,sample_size=8,epochs=20,label_smooth=1.0,p_flip = 0,checkpoints='model.model',recon='/.images'):
        self.device = device
        self.loader = loader
        self.G = G
        self.D = D
        self.optimG = optimG
        self.optimD = optimD
        self.path = checkpoints 
        self.recon = recon 
        self.g_losses = []
        self.d_losses = []
        self.epochs = epochs
        self.start_epoch = 0
        self.label_smooth = label_smooth
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.p_flip = p_flip
        self.test_noise = 0

    def save_model(self, epoch):
        print("Saving Model at '{}'".format(self.path))
        model = {
                'epoch': epoch+1,
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
            self.start_epoch = check['epoch']
            self.g_losses = check['g_losses']
            self.d_losses = check['d_losses']
            self.G.load_state_dict(check['generator'])
            self.D.load_state_dict(check['discriminator'])
        except:
            print("Model could not be loaded from {}. Training from Scratch".format(self.path))
            self.start_epoch= 0
            self.g_losses = []
            self.d_losses = []

    def sample_images(self,epoch,nrow=8):
        with torch.no_grad():
            images = self.G(self.test_noise)
            img = torchvision.utils.make_grid(images)
            torchvision.utils.save_image(images,"%s/epoch%d.png" % (self.recon,epoch+1),nrow=nrow)

    def train(self):
        self.G.train()
        self.D.train()
        for epoch in range(self.start_epoch, self.epochs+1):
            print("Epoch %d of %d" % (epoch+1,self.epochs))
            running_G = 0.0
            running_D = 0.0
            for i, data in enumerate(self.loader, 1):
                images,labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                loss_d,loss_g = self.train_loop(images,labels)
                running_G += loss_g
                running_D += loss_d
                self.g_losses.append(loss_g)
                self.d_losses.append(loss_d)
            self.save_model(epoch)
            self.G.eval()
            self.D.eval()
            print("Epoch {} : Mean Generator Loss {} Mean Discriminator Loss {}".format(epoch+1,running_G/i,running_D/i))      
            print("Sampling and saving images")
            self.sample_images(epoch)
            self.G.train()
            self.D.train()

