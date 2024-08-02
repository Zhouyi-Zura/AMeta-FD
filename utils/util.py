from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-10)

class SSI_Loss(nn.Module):
    def __init__(self):
        super(SSI_Loss, self).__init__()

    def forward(self, img_ori, img_p):
        img_ori = normalization(img_ori)
        img_p = normalization(img_p)
        mean_ori = torch.mean(img_ori)
        std_ori = torch.std(img_ori)
        mean_p = torch.mean(img_p)
        std_p = torch.std(img_p)
        ssi = (std_ori * mean_p) / (mean_ori * std_p)

        return ssi

