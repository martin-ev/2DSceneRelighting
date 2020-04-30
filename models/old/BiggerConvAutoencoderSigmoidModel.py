# inspired from https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c

import torch.nn as nn

class Autoencoder(nn.Module):    
    def __init__(self):
        super(Autoencoder,self).__init__()        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(16,32,kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(32,64,kernel_size=5),
            nn.ReLU(True))        
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(64,32,kernel_size=5),
            nn.ReLU(True),         
            nn.ConvTranspose2d(32,16,kernel_size=5),
            nn.ReLU(True),    
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.Sigmoid())
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x