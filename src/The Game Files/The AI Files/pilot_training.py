#Kody Graham
#11/03/2025

#Will contain the training class for my agent

import os, math,time
import numpy as np
import torch
import torch.nn as nn

#Model
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=5, hidden=64, act_dim =2):
        super().__init__()

def gae_return(rews, vals, dones, gamma=.995, lam=.95):


def batchify()

def train()
