import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim



# Get the available device

if torch.cuda.is_available():
    dev = "cuda:0"  
else:
    dev = "cpu"
device = torch.device(dev)


# buils autoencoder consisting from encoder and decoder 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder is a feed-forward network with input size 784x32 
        # activation function is Relu 
        self.encoder = nn.Sequential(
            nn.Linear(784, 32),
            nn.ReLU(True))
        
        # decoder is a feed-forward network 
        # it needs to return the same sized images as inputs 
        # activation function is Sigmoid 
        self.decoder = nn.Sequential(
            nn.Linear(32, 784),
            nn.Sigmoid())
        
    # performs forward pass through autoencoder 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


# add Gaussian Noise to given image 
class AddGaussianNoise(object):
    # n_f is a noise factor 
    def __init__(self, n_f):
        self.n_f = n_f
    
    # add noise to images 
    def forward(self, tensor):
        return torch.clamp((tensor + torch.randn(tensor.size()) * self.n_f), 0, 1)

# reshapes image size 
def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x  



# specifies transformation of downloaded images 
# convert them to tensors 
transform = transforms.Compose([transforms.ToTensor()])

#download training and testing data 
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
testset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

# pass data to trainloader with specific batch sizes and shuffling options 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=False)

# noise factor to add Gaussian noise to images 
n_f =0.1
num_epochs=10
model = Autoencoder()
model = model.to(device)
add_noise = AddGaussianNoise(n_f)

# specify loss: mean square error loss
criterion = nn.BCELoss() 
# set optimizer and learning rate 
optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01, 
                                 weight_decay=1e-5)

outputs = []
for epoch in range(num_epochs):
    for data in trainloader:
        orig_img, _ = data
        orig_img = orig_img.view(-1, 784)
        noisy_img = add_noise.forward(orig_img)
        noisy_img = noisy_img.to(device)
        recon_img = model(noisy_img)
        orig_img = orig_img.to(device)
        loss = criterion(recon_img, orig_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
    if epoch % 10 == 0:
        for data in testloader:
            orig_img, _ = data
            orig_img = orig_img.view(-1, 784)
            noisy_img = add_noise.forward(orig_img)
            noisy_img = noisy_img.to(device)
            recon_img = model(noisy_img)
            x_hat = to_img(recon_img.cpu().data)
            x_noisy = to_img(noisy_img.cpu().data)
            outputs.append((epoch, x_noisy, x_hat),)
            


plt.figure(figsize=(10, 2))
for k in range(0, 10):
    
    imgs = outputs[k][2].detach().numpy()
    recon = outputs[k][1].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])
        
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item[0])

plt.savefig("autoencoder0.1.png")



