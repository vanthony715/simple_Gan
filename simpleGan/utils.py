#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 19:23:39 2021

@author: avasquez
"""
import os
import shutil
import torch
import torchvision.utils as vutils

def clearFolder(Path):
    if os.path.isdir(Path):
        print('Removing File: ', Path)
        shutil.rmtree(Path)
    print('Creating File: ', Path)
    os.mkdir(Path)
    
def saveNet(OutPath, Network, Epoch, NetName):
        torch.save(Network.state_dict(), 
                   os.path.join(OutPath, NetName + '_{}.pth'. format(Epoch)))
        
def saveImage(OutPath, Sample, Epoch, Name):
    SavePath = OutPath + Name + str(Epoch) + '.png'
    vutils.save_image(Sample, SavePath, normalize = True)