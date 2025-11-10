#Kody Graham
#11/03/2025

#Will contain the training class for my agent

#Note for self: Done

from __future__ import annotations
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from game_env import GameEnv

#Model, network shared body: policy head and value head
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int =5, hidden: int=128, act_dim: int =2):
        super().__init__() #Initialize nn module
        self.body = nn.Sequential( #Extract shared features
            nn.Linear(obs_dim, hidden), nn.Tanh(), #First linear layer maps the obs to hidden
            nn.Linear(hidden, hidden), nn.Tanh(), #Second hidden layer
        )
        self.pi = nn.Linear(hidden, act_dim) #Policy head
        self.v = nn.Linear(hidden, 1) #Value head

        #Match my autopilot_torch
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.pi.weight, gain=.01) #Small init for policy
        nn.init.constant_(self.pi.bias, 0.0)
        nn.init.orthogonal_(self.v.weight, gain=1.0) #For value head
        nn.init.constant_(self.v.bias, 0.0)

    #To pass forward through the network
    def forward(self, x: torch.Tensor):
        h= self.body(x) #Shared features
        return self.pi(h), self.v(h)

#Compute GAE advantages and returns
def gae_return(rewards, values, dones, gamma=.99, lam=.95, last_val = 0.0):

    T = rewards.shape[0] #Num time steps in rollout
    device = rewards.device

    values_ext = torch.cat([values, torch.as_tensor([last_val], device=device)])
    adv= torch.zeros(T, device=device) #Buffer for advantages
    lastgaelam = 0.0

    for t in reversed(range(T)): #Work backwards
        non_terminal= 1.0- dones[t]
        delta = rewards[t] + gamma * values_ext[t+1] * non_terminal - values_ext[t] #td resid
        lastgaelam = delta +gamma * lam * non_terminal *lastgaelam #GAE formula
        adv[t]= lastgaelam

    retrn = adv+values
    return adv, retrn

#Mini batch generator with random shuffle
def batchify(bs: int, *arrays):
    n= arrays[0].shape[0]
    idx= np.arange(n)
    np.random.shuffle(idx)
    for start in range(0,n,bs):
        j = idx[start:start+bs]
        yield tuple(a[j] for a in arrays) #Yield aligned mini batches across input arrays

#Evaluate current policy greedily
def evaluate_policy(model: ActorCritic, device, episodes: int=25):

    env = GameEnv()
    model.eval()

    total_len =0.0
    total_pipes = 0.0

    for episode in range(episodes):
        obs, _ = env.reset(seed=1234+ episode)
        ep_len = 0
        ep_pipes = 0

        while True: #Run until termination or truncation
            x= torch.tensor(obs,dtype=torch.float32, device=device).unsqueeze(0) #Convert obj to tensor
            logits, _ = model(x)
            act = int(torch.argmax(logits, dim=1).item())
            obs, r, terminated, truncated, info = env.step(act)

            ep_len +=1
            if info.get("passed", False):
                ep_pipes+=1
            if terminated or truncated:
                break

        total_len += ep_len
        total_pipes+=ep_pipes

    model.train()
    return total_len / episodes, total_pipes/ episodes #Averages for episodes

#Main training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Gpu if available
    env= GameEnv() #Instantiate the environment

    observation_dim, action_dim =5,2

    #Build AC model and move to device
    model = ActorCritic(obs_dim=observation_dim, hidden=128, act_dim=action_dim).to(device)

    #Defaults for my MLP
    total_updates=100000
    steps_per_roll = 4096
    mini_batch_size = 512
    ppo_epochs = 5

    initial_lr = 3e-4 #lr = learning rate, throughout my class
    final_lr = 3e-5

    clip_eps = 0.2
    vf_coef = 0.5
    entropy_coef_start = 0.02
    entropy_coef_end = .002
    target_kl= .02 #kl= divergence for early stop, also throughout class
    max_grad_norm = 0.7

    opt = optim.Adam(model.parameters(), lr=initial_lr, eps=1e-5) #Adam optimizer for model parameters

    best_eval_pipes = -1.0 #Track best eval
    save_path= os.path.join(os.path.dirname(__file__),"autopilot_policy.pt") #Save best model

    print("#$#$#$# Stop the program at anytime and the current policy will be saved and training will end. #$#$#$#")
    print("#$#$#$# Note: Each update = 100 Episodes. #$#$#$#")

    try:
        # PPO outer loop over updates
        for update in range(1,total_updates+1):
            progress = (update-1)/ max(1, total_updates-1)
            learning_rate_now = initial_lr + (final_lr - initial_lr) * progress
            for pg in opt.param_groups:
                pg["lr"] = learning_rate_now
            ent_coef= entropy_coef_start + (entropy_coef_end - entropy_coef_start) * progress

            #Buffers for rollout data
            obs_buf, act_buf=[], []
            logp_buf, reward_buf = [], []
            done_buf, val_buf=[], []
            ep_lens, ep_rets = [],[] #Episode length and reward for my log

            obs, _ = env.reset(seed=update) #Reset environment and get the initial observation
            ep_reward = 0.0 #Running return for episode
            ep_len = 0 #Running length for episode

            with torch.no_grad():#On policy no gradients
                for _ in range(steps_per_roll):
                    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) #MAke batch
                    logits, value = model(x) #Pass forward
                    dist = Categorical(logits=logits) #Distribution over actions
                    action = dist.sample()
                    logp = dist.log_prob(action) #Logrithmic probability of sampled action

                    next_obs, reward, terminated, truncated, info = env.step(int(action.item())) #Step env with action
                    done = terminated or truncated #Flag end episode

                    obs_buf.append(np.asarray(obs, dtype=np.float32))
                    act_buf.append(int(action.item()))
                    logp_buf.append(float(logp.item()))
                    reward_buf.append(float(reward))
                    done_buf.append(float(done))
                    val_buf.append(float(value.item())) #Predicted value

                    ep_reward += float(reward)
                    ep_len += 1
                    obs = next_obs

                    #Log stats when done with episode
                    if done:
                        ep_lens.append(ep_len)
                        ep_rets.append(ep_reward)
                        ep_reward = 0.0
                        ep_len = 0
                        obs, _ = env.reset()

            #Bootstrap val for final state of rollout
            with torch.no_grad():
                if done_buf and done_buf[-1] == 0.0:
                    x_last = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    _, last_val_t = model(x_last)
                    last_val = float(last_val_t.item())
                else:
                    last_val=0.0

            #Convert buffers to tensor on device
            obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
            act_t =torch.tensor(np.array(act_buf), dtype=torch.int64, device=device)
            old_logp_t= torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
            reward_t= torch.tensor(np.array(reward_buf), dtype=torch.float32, device=device)
            done_t= torch.tensor(np.array(done_buf), dtype=torch.float32, device=device)
            val_t =torch.tensor(np.array(val_buf), dtype=torch.float32, device=device)

            #Compute GAE returns and advantages
            adv_t, ret_t = gae_return(reward_t, val_t, done_t, gamma=.99, lam=.95, last_val=last_val,)
            adv_t = (adv_t- adv_t.mean()) / (adv_t.std() + 1e-8) #Normalize

            #Update PPO
            for _ in range(ppo_epochs):
                kl_exceeded = False

                #Loop over mini batches
                for (b_obs, b_act, b_old_logp, b_adv, b_ret, b_val_old,) \
                        in batchify(mini_batch_size, obs_t.cpu().numpy(),
                                    act_t.cpu().numpy(), old_logp_t.cpu().numpy(),
                                    adv_t.cpu().numpy(), ret_t.cpu().numpy(),
                                    val_t.cpu().numpy()): #Draw mini batches

                    #Convert mini batch nack to tensor on device
                    b_obs = torch.tensor(b_obs, dtype=torch.float32, device=device)
                    b_act = torch.tensor(b_act, dtype=torch.int64, device=device)
                    b_old_logp = torch.tensor(b_old_logp, dtype=torch.float32, device=device)
                    b_adv = torch.tensor(b_adv, dtype=torch.float32, device=device)
                    b_ret = torch.tensor(b_ret, dtype=torch.float32, device=device)
                    b_val_old = torch.tensor(b_val_old, dtype=torch.float32, device=device)

                    logits, value = model(b_obs) #Forward
                    dist = Categorical(logits=logits) #My policy distribution
                    logp = dist.log_prob(b_act)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(logp - b_old_logp)
                    pg_loss1= ratio*b_adv
                    pg_loss2= torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps)*b_adv
                    pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                    value_pred = value.squeeze(-1)
                    v_pred_clipped = b_val_old + (value_pred - b_val_old).clamp(-.2,.2) #Clipped prediction
                    vf_loss1 = (b_ret - value_pred).pow(2) #MSE unclipped
                    vf_loss2 = (b_ret - v_pred_clipped).pow(2) #MSE clipped
                    v_loss = .5* torch.max(vf_loss1, vf_loss2).mean() #Value loss

                    #Total PPO loss
                    loss = pg_loss + vf_coef * v_loss- ent_coef * entropy

                    opt.zero_grad() #Clear
                    loss.backward() #Through network
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step() #Update

                    #Compute kl divergence for stability checking
                    with torch. no_grad():
                        approx_kl = (b_old_logp- logp).mean().item()
                    if approx_kl > target_kl: #stop if change is too big
                        kl_exceeded = True
                        break

                #Early exit if policy is plateaued
                if kl_exceeded:
                    break

            avg_len = float(np.mean(ep_lens)) if ep_lens else 0.0 #Average episode len across rollout
            avg_ret = float(np.mean(ep_rets)) if ep_rets else 0.0 #Average episode return across rollout

            #Display statistics per update and per 10 updates
            if update % 10==0:
                eval_len, eval_pipes = evaluate_policy(model, device, episodes=100) #Episodes is per update
                if eval_pipes > best_eval_pipes: #If new best agent
                    best_eval_pipes = eval_pipes
                    torch.save(model.state_dict(), save_path) #Save weights

                print(f"Update {update:05d} | train_length = {avg_len:6.1f} | train_return = {avg_ret:+.2f} "
                      f"| eval_length = {eval_len:6.1f} | eval_pipes = {eval_pipes:5.2f} "
                      f"| learning_rate_now = {learning_rate_now:.6f} ent = {ent_coef:.4f}")
            else:
                print(f"Update {update:05d} | train_length = {avg_len:6.1f} | train_return = {avg_ret:+.2f} "
                        f"| ent = {ent_coef:.4f} | learning_rate_now = {learning_rate_now:.6f} ")

    #Quick escape
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt recievied. Saving the current model...")
        torch.save(model.state_dict(), save_path) #Save current weights
        print("Training Saved:", save_path)
        return

    if not os.path.isfile(save_path):
        torch.save(model.state_dict(), save_path)

    #Final
    print("Final best_eval_pipes:", best_eval_pipes)
    print("Saved:", save_path)

#Run it
if __name__ == "__main__":
    train()



