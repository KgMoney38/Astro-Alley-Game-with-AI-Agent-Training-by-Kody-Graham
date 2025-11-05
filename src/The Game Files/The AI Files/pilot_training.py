#Kody Graham
#11/03/2025

#Will contain the training class for my agent

import os, math,time
import numpy as np
import torch
import torch.nn as nn

#Model, network shared body: policy head and value head
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=5, hidden=64, act_dim =2):
        super().__init__() #Initialize nn module
        self.body = nn.Sequential( #Extract shared features
            nn.Linear(obs_dim, hidden), nn.Tanh(), #First linear layer maps the obs to hidden
            nn.Linear(hidden, hidden), nn.Tanh(), #Second hidden layer
        )
        self.pi = nn.Linear(hidden, act_dim) #Policy head
        self.v = nn.Linear(hidden, 1) #Value head

    #To pass forward through the network
    def forward(self, x):
        h= self.body(x) #Shared features
        return self.pi(h), self.v(h)

#Compute GAE advantages and returns
def gae_return(rews, vals, dones, gamma=.995, lam=.95):

    T = len(rews) #Num time steps in rollout
    adv= torch.zeros(T, dtype=torch.float32) #Buffer for advantages
    lastgaelam = 0
    for t in reversed(range(T)): #Work backwards
        next_non_terminal = 1- float(dones[t])
        next_value= 0 if t == T-1 else float(vals[t+1])
        delta = float(rews[t]) + gamma* lam * next_non_terminal* lastgaelam #td resid
        lastgaelam = delta +gamma * lam * next_non_terminal *lastgaelam
        adv[t]= lastgaelam
    ret = adv+vals
    return adv, ret

#Small batch generator with random shuffle
def batchify(*arrays,bs):
    n= arrays[0].shape[0]
    idx= np.arange(n)
    np.random.shuffle(idx)
    for start in range(0,n,bs):
        j = idx[start:start+bs]
        yield (a[j] for a in arrays) #Yield aligned batches across input arrays

def train():

