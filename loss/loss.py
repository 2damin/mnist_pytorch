import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys

class MNISTloss(nn.Module):
    def __init__(self, device = torch.device("cpu")):
        super(MNISTloss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self, out, gt):
        self.loss(out, gt)
        return self.loss(out, gt)

def get_criterion(crit = "mnist", device = torch.device("cpu")):
    if crit is "mnist":
        return MNISTloss(device = device)