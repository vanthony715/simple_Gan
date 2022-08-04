#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:34:21 2021

@author: avasquez
"""

import os, gc, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import utils

##clear bins
torch.cuda.empty_cache()
gc.collect()

##start clock
tic = time.time()

if __name__ == "__main__":
    ##Paths
    basepath = '/home/domain/avasquez/data/Neuro/' ##path to dataset
    dataPath = basepath + 'chips/chipsSemi_06_26_2021/' ##path to chips
    outPath = basepath + 'chips/output/' ##path to output like weights and images
    
    CUDA = True
    BATCH_SZ = 16
    IMG_CHANNEL = 3
    Z_DIM = 100
    G_HID = 64
    X_DIM = 64
    D_HID = 64
    EPOCH_NUM = 10
    REAL_LABEL = 1
    FAKE_LABEL = 0
    LR = 2e-4
    seed = 1
    
    ##clear outpath and create directory if it doesn't exist
    utils.clearFolder(outPath)
    
    ##GPU Resource Conf
    CUDA = CUDA and torch.cuda.is_available()
    print('Pytorch Version: ', torch.__version__)
    if CUDA:
        print('CUDA version: ', torch.version.cuda)
    if seed is None:
        seed = np.random.randint(1, 10000)
    print('Random Seed: ', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if CUDA else "cpu")
    
    ##generator network
    class GNet(nn.Module):
        def __init__(self):
            super(GNet, self).__init__()
            self.main = nn.Sequential(
                ##layer 1
                nn.ConvTranspose2d(Z_DIM, G_HID * 8, 4, 1, 0, bias = False),
                nn.BatchNorm2d(G_HID * 8), nn.ReLU(True),
                ##layer 2
                nn.ConvTranspose2d(G_HID * 8, G_HID * 4, 4, 2, 1, bias = False),
                nn.BatchNorm2d(G_HID * 4), nn.ReLU(True),
                ##layer 3
                nn.ConvTranspose2d(G_HID * 4, G_HID * 2, 4, 2, 1, bias = False),
                nn.BatchNorm2d(G_HID * 2), nn.ReLU(True),
                ##layer 4
                nn.ConvTranspose2d(G_HID * 2, G_HID , 4, 2, 1, bias = False),
                nn.BatchNorm2d(G_HID), nn.ReLU(True),
                ##output layer
                nn.ConvTranspose2d(G_HID, IMG_CHANNEL, 4, 2, 1, bias = False),
                nn.Tanh())
        def forward(self, input):
            return self.main(input)
        
    ##discriminator network
    class DNet(nn.Module):
        def __init__(self):
            super(DNet, self).__init__()
            self.main = nn.Sequential(
                ##layer 1
                nn.Conv2d(IMG_CHANNEL, D_HID, 4, 2, 1, bias = False),
                nn.LeakyReLU(0.2, inplace = True),
                ##layer 2
                nn.Conv2d(D_HID, D_HID * 2, 4, 2, 1, bias = False),
                nn.BatchNorm2d(D_HID * 2),
                nn.LeakyReLU(0.2, inplace = True),
                ##layer 3
                nn.Conv2d(D_HID * 2, D_HID * 4, 4, 2, 1, bias = False),
                nn.BatchNorm2d(D_HID * 4),
                nn.LeakyReLU(0.2, inplace = True),
                ##layer 4
                nn.Conv2d(D_HID * 4, D_HID * 8, 4, 2, 1, bias = False),
                nn.BatchNorm2d(D_HID * 8),
                nn.LeakyReLU(0.2, inplace = True),
                ##output layer
                nn.Conv2d(D_HID * 8, 1, 4, 1, 0, bias = False),
                nn.Sigmoid())
        def forward(self, input):
            return self.main(input).view(-1, 1).squeeze(1)
            
    ##helper function
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    ## instantiate generator
    gNet = GNet().to(device)
    gNet.apply(weights_init)
    ## instantiate discriminator
    dNet = DNet().to(device)
    dNet.apply(weights_init)
    
    print(gNet)
    print(dNet)
    
    ## define loss functions
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(dNet.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizerG = optim.Adam(gNet.parameters(), lr=LR, betas=(0.5, 0.999))
    
    ##Custom data
    dataset = dset.ImageFolder(root=dataPath, 
                             transform=transforms.Compose([
                                 transforms.Resize(X_DIM),
                                 transforms.CenterCrop(X_DIM),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5)),
                             ]))
    
    ##define data loader                              
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SZ, shuffle=True, num_workers=4)
    
    ##TRAIN
    visNoise = torch.randn(BATCH_SZ, Z_DIM, 1, 1, device = device)
    for epoch in range(EPOCH_NUM):
        for i, data in enumerate(dataloader):
            xReal = data[0].to(device)
            realLabel = torch.full((xReal.size(0),), REAL_LABEL, device = device)
            fakeLabel = torch.full((xReal.size(0),), FAKE_LABEL, device = device)
            ##update D with real data
            dNet.zero_grad()
            yReal = dNet(xReal)
            realLossD = criterion(yReal, realLabel)
            realLossD.backward()
            # ##update D with fake data
            zNoise = torch.randn(xReal.size(0), Z_DIM, 1, 1, device=device)
            xFake = gNet(zNoise)
            yFake = dNet(xFake.detach())
            loss_D_fake = criterion(yFake, fakeLabel)
            loss_D_fake.backward()
            optimizerD.step()
            # ##update G with fake data
            gNet.zero_grad()
            yFake_r = dNet(xFake)
            loss_G = criterion(yFake_r, realLabel)
            loss_G.backward()
            optimizerG.step()
            
            ##save images and network
            if i % 100 == 0:
                print('Epoch {} [{}/{}] realLossD: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(
                    epoch, i, len(dataloader), realLossD.mean().item(), loss_D_fake.mean().item(), loss_G.mean().item()))
                
            if i % 100 == 0:
                if i < 200:
                    vutils.save_image(xReal, os.path.join(outPath, 'realSamples.png'), normalize = True)
                
                with torch.no_grad():
                    viz_sample = gNet(visNoise)
                    utils.saveImage(outPath, viz_sample, epoch, 'fakeSamples')
                    utils.saveNet(outPath, gNet, epoch, 'gNet')
                    utils.saveNet(outPath, dNet, epoch, 'dNet')
            
    gc.collect()
    ##Clock time
    print('\n----Time----\n')
    toc = time.time()
    tf = round((toc - tic), 1)
    print('Time to Run (s): ', tf)