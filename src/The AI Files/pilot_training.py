#Kody Graham
#11/03/2025

#Will contain the training class for my agent

#Note for self: Done

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from game_env import GameEnv

#Model, network shared body: policy head and value head
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int =5, hidden: int=64, act_dim: int =2):
        super().__init__() #Initialize nn module
        self.body = nn.Sequential( #Extract shared features
            nn.Linear(obs_dim, hidden), nn.Tanh(), #First linear layer maps the obs to hidden
            nn.Linear(hidden, hidden), nn.Tanh(), #Second hidden layer
        )
        self.pi = nn.Linear(hidden, act_dim) #Policy head
        self.v = nn.Linear(hidden, 1) #Value head

    #To pass forward through the network
    def forward(self, x: torch.Tensor):
        h= self.body(x) #Shared features
        return self.pi(h), self.v(h)

#Compute GAE advantages and returns
def gae_return(rewards, vals, dones, gamma=.995, lam=.95):

    T = len(rewards) #Num time steps in rollout
    adv= torch.zeros(T, dtype=torch.float32, device = vals.device) #Buffer for advantages
    lastgaelam = 0.0

    for t in reversed(range(T)): #Work backwards
        nextnonterminal= 1.0- float(dones[t])
        nextvalue = 0.0 if t == T-1 else float(vals[t+1])
        delta = float(rewards[t]) + gamma * nextvalue * nextnonterminal - float(vals[t]) #td resid
        lastgaelam = delta +gamma * lam * nextnonterminal *lastgaelam
        adv[t]= lastgaelam
    _return = adv+vals
    return adv, _return

#Mini batch generator with random shuffle
def batchify(*arrays,bs):
    n= arrays[0].shape[0]
    idx= np.arange(n)
    np.random.shuffle(idx)
    for start in range(0,n,bs):
        j = idx[start:start+bs]
        yield tuple(a[j] for a in arrays) #Yield aligned mini batches across input arrays

#Main training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Gpu if available
    env= GameEnv() #Instantiate the environment

    observation_dim=5
    action_dim=2

    #Build AC model and move to device
    model = ActorCritic(obs_dim=observation_dim, hidden=64, act_dim=action_dim).to(device)
    opt= optim.Adam(model.parameters(), lr=3e-4, eps=1e-5) #Adam optimizer

    #Defaults for my MLP
    total_updates=333
    steps_per_roll = 4096
    mini_batch_size = 512
    ppo_epochs = 4
    clip_eps = 0.2
    vf_coef = 0.5
    entropy_coef_start = 0.01
    entropy_coef_end = .003

    best_avg_len= -1.0
    best_avg_ret= -1.0
    save_path= os.path.join(os.path.dirname(__file__),"autopilot_policy.pt") #Save best checkpoint

    #PPO outer loop over updates
    for update in range(1,total_updates+1):
        #Cosine decay LR and entropy
        progress = (update-1)/max(1,total_updates-1)
        lr_now = 3e-4*0.5*(1.0 + math.cos(math.pi * progress))
        for pg in opt.param_groups:
            pg["lr"] = lr_now #apply to ALL param groups
        ent_coef = entropy_coef_end + (entropy_coef_start-entropy_coef_end)*(1.0-progress) #Linear decay

        #Buffers
        obs_buf=[]
        act_buf=[]
        logp_buf=[]
        reward_buf=[]
        done_buf=[]
        val_buf=[]

        lens = [] #Episode len for my log
        returns= [] #Episode total reward for logging

        obs, _ = env.reset() #Reset environment and get the initial observation
        ep_reward = 0.0 #Running return for episode
        ep_len = 0 #Running length for episode

        with torch.no_grad():#On policy no gradients
            for _ in range(steps_per_roll):
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) #MAke batch
                logits, value = model(x) #Pass forward
                dist = Categorical(logits=logits) #Distribution over actions
                action = dist.sample()
                logp = dist.log_prob(action) #Logrithmic probability of sampled action

                next_obs, reward, terminated, truncated, _ = env.step(int(action.item())) #Step
                done = terminated or truncated

                obs_buf.append(obs)
                act_buf.append(int(action.item()))
                logp_buf.append(float(logp.item()))
                reward_buf.append(float(reward))
                done_buf.append(float(1.0 if done else 0.0))
                val_buf.append(float(value.item()))

                ep_reward += float(reward)
                ep_len += 1
                obs = next_obs

                #Log stats when done
                if done:
                    lens.append(ep_len)
                    returns.append(ep_reward)
                    ep_reward = 0.0
                    ep_len = 0
                    obs, _ = env.reset()

            #Bootstrap val for final state of rollout
            #if len(done_buf)> 0 and done_buf[-1] == 0.0:
            #    x_last = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            #    _, last_val_t = model(x_last)
            #    last_val = float(last_val_t.item())
            #else:
             #   last_val=0.0



        #Convert buffers to tensor on device
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t =torch.tensor(np.array(act_buf), dtype=torch.int64, device=device)
        old_logp_t= torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        reward_t= torch.tensor(np.array(reward_buf), dtype=torch.float32, device=device)
        done_t= torch.tensor(np.array(done_buf), dtype=torch.float32, device=device)
        val_t =torch.tensor(np.array(val_buf), dtype=torch.float32, device=device)

        #Compute GAE returns and advantages
        adv_t, ret_t = gae_return(reward_t, val_t, done_t, gamma=.995, lam=.95)
        adv_t = (adv_t- adv_t.mean()) / (adv_t.std() + 1e-8)

        #Update PPO
        for _ in range(ppo_epochs):
            for b_obs, b_act, b_old_logp, b_adv, b_ret in batchify(obs_t, act_t, old_logp_t, adv_t, ret_t, bs=mini_batch_size): #Draw mini batches

                logits, value = model(b_obs) #Forward
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - b_old_logp)
                pg_loss1= ratio*b_adv
                pg_loss2= torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps)*b_adv
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                v_loss = .5* (b_ret-value.squeeze(-1)).pow(2).mean()
                loss = pg_loss + vf_coef*v_loss- ent_coef*entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        avg_len = float(np.mean(lens)) if lens else 0.0 #Average episode len across rollout
        avg_ret = float(np.mean(returns)) if returns else 0.0 #Average episode return across rollout
        print(f"Update {update:03d} | avg_ep_length={avg_len:5.1f} | avg_ep_return={avg_ret:+.2f}")

        #Save best checkpoint thus far
        if avg_len > best_avg_len:
            best_avg_len = avg_len
            torch.save(model.state_dict(), save_path)
        if avg_ret > best_avg_ret:
            best_avg_ret = avg_ret

    if not os.path.isfile(save_path):
        torch.save(model.state_dict(), save_path)

    #Final
    print(f"Highest avg length: {best_avg_len:+.2f}", f"\nHighest avg return: {best_avg_ret:+.2f}")
    print("Saved:", save_path)

#Run it
if __name__ == "__main__":
    train()



