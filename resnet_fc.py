import numpy as np
import matplotlib.pyplot as plt

import sys
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import models


# Get the available device

if torch.cuda.is_available():
    dev = "cuda:0"  
else:
    dev = "cpu"
device = torch.device(dev)

# multi-class classification model based on ResNet 
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # set pretrained ResNet18
        self.model_resnet = models.resnet18(pretrained=True)
        
        # do not train ResNet 
        for param in self.model_resnet.parameters():
            param.requires_grad = False
        
        # obtain input dimentions to the last Fully Connected Layer of ResNet 
        fc_inputs = self.model_resnet.fc.in_features
        
        # replace the last FC layer with identity 
        self.model_resnet.fc = nn.Identity()
        
        # create a new part of the model 
        self.labels_model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes), 
        )

    def forward(self, images):
        
        # forward pass through ResNet18 and the new created model 
        # we should not optimize over ResNet parameters 
        with torch.no_grad():
            features = self.model_resnet(images)
            
        labels = self.labels_model(features)
        return labels


# performs training 
def train_model(model, criterion, optimizer, scheduler, epochs=25):
   
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        print("Epoch:", epoch)
        
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0 
            
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # reset gradients 
                optimizer.zero_grad()
                # compute loss 
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # update weights if training 
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == "train":
                scheduler.step()
            
            epoch_loss = running_loss / datasizes[phase]
            epoch_acc = running_corrects.double()/datasizes[phase]
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                test_losses.append(epoch_loss)
            
            
    np.savetxt("train_losses_fc_tr.csv", train_losses)
    np.savetxt("test_losses_fc_tr.csv", test_losses)
    return model



# performs data augmentation: horizontal flip, vertical flip
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

# normalizes sthe test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])


# download training and testind data and load it to loader 
trainset = torchvision.datasets.ImageFolder(root='/kaggle/input/Birds/180/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=0, shuffle=True)
testset = torchvision.datasets.ImageFolder(root='/kaggle/input/Birds/180/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=0, shuffle=False)

dataloaders = {"train": trainloader,"test": testloader}
datasizes = {"train": len(trainset),"test": len(testset)}
cl = list(trainset.class_to_idx.keys())
# number of total classes 
n_cl = len(cl)

model = CNN(n_cl)
# set loss 
criterion = nn.CrossEntropyLoss()
# define optimizer as Stochastic Gradient Decsent 
optimizer_ft = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
exp_lr_sc = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model = model.to(device)
model_ft = train_model(model, criterion, optimizer_ft, exp_lr_sc, epochs=25)


