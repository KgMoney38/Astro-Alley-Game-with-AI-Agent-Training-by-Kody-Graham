#Kody Graham
#11/03/2025

#Will contain the training class for my agent

#Note for self: Done

from __future__ import annotations
import os, math, copy
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from torch.distributions.categorical import Categorical

from game_env import GameEnv

#non block keyboard check
try:
    import msvcrt
    HAVE_MSVCRT = True
except ImportError:
    HAVE_MSVCRT = False

#Model, network shared body: policy head and value head, changed my 2 x128 tanh MLP to a 3x256
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int =7, hidden: int=256, act_dim: int =2):
        super().__init__() #Initialize nn module
        self.body = nn.Sequential( #Extract shared features
            nn.Linear(obs_dim, hidden), nn.Tanh(), #First linear layer maps the obs to hidden
            nn.Linear(hidden, hidden), nn.Tanh(), #Second layer
            nn.Linear(hidden, hidden), nn.Tanh(),  #Third layer

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

def batchify_torch(bs: int, *tensors: torch.Tensor):
    n= tensors[0].shape[0]
    idx = torch.randperm(n, device= tensors[0].device)
    for start in range(0, n, bs):
        j = idx[start:start+bs]
        yield tuple(t.index_select(0, j) for t in tensors)

#Evaluate current policy greedily
def evaluate_policy(model: ActorCritic, device, episodes: int=50):

    env = GameEnv(domain_randomize=False)
    model.eval()

    total_len =0.0
    total_pipes = 0.0

    for episode in range(episodes):
        obs, _ = env.reset(seed=1234+ episode)
        ep_len = 0
        ep_pipes = 0

        while True: #Run until termination or truncation
            with torch.inference_mode():
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

#Format the elapsed seconds to the nearest 5
def format_elapsed(seconds: float)-> str:
    if seconds < 0:
        seconds = 0
    rounded = int(round(seconds/5.0)*5)
    h = rounded //3600
    m = (rounded % 3600) //60
    s = rounded % 60
    if h > 0:
        return f"{h:01d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"

#Main training loop
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Gpu if available
    env= GameEnv(domain_randomize=False) #Instantiate the environment

    observation_dim, action_dim = env.obs_dim, 2

    #Build AC model and move to device
    model = ActorCritic(obs_dim=observation_dim, hidden=256, act_dim=action_dim).to(device)

    import copy
    INFER_ON_CPU = True
    if INFER_ON_CPU:
        model_rollout = copy.deepcopy(model).to("cpu").eval()
    else:
        model_rollout = model

    #Training graphs
    plt.ion()
    plt.style.use("dark_background")

    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    fig.patch.set_facecolor("black")

    ax_tlen, ax_tret = axs[0]
    ax_eval_len, ax_eval_pipes = axs[1]

    def style_axis():
        for ax in (ax_tlen, ax_tret, ax_eval_len, ax_eval_pipes):
            ax.set_facecolor("black")
            ax.grid(True, alpha=0.25, color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1.5)

        #Labels
        ax_tlen.set_ylabel("Train Length", color="white")
        ax_tret.set_ylabel("Train Return", color="white")
        ax_eval_len.set_ylabel("Eval Length", color="white")
        ax_eval_pipes.set_ylabel("Eval Pipes Avg Estimate", color="white")

        ax_tlen.set_xlabel("Time (minutes)", color="white")
        ax_tret.set_xlabel("Time (minutes)", color="white")
        ax_eval_len.set_xlabel("Update #", color="white")
        ax_eval_pipes.set_xlabel("Update #", color="white")

        #Top row
        for ax in (ax_tlen, ax_tret):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        #Bottom
        for ax in (ax_eval_len, ax_eval_pipes):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    style_axis()

    #My Console Mirror Section
    from collections import deque
    from matplotlib import transforms as _mtransforms
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    #Layout
    LEFT_F = 0.07  #Left Margin
    RIGHT_F = 0.98  #Right Margin
    TOP_F = 0.90  #Top
    GAP_ABOVE_CONSOLE_F = 0.08  #Between charts

    CONSOLE_H_F = 0.26  #Console Height
    BOTTOM_F = 0.06  #Bot gap

    #Text
    CONSOLE_FONT = 9
    LINE_PX = int(CONSOLE_FONT * 1.60)
    PAD_TOP_PX = 10
    PAD_BOT_PX = 10
    PAD_LEFT_PX = 10
    RULE_GAP_PX = 10

    #Instructions
    INSTR_LINES = [
        "***** Stop the program at anytime and the best policy so far will be saved and training will end. ******",
        "***** Press 'Q' to stop training and save best model. Note: Focus must be in the graph window*****",
        "***** Each update = 50 Episodes. Note: Press ESC to exit fullscreen.***** ",
    ]
    INSTR_HEIGHT_PX = len(INSTR_LINES) * LINE_PX

    #Vertical space for the console and gap for the charts above it.
    plt.subplots_adjust(
        left=LEFT_F, right=RIGHT_F, top=TOP_F,
        bottom=BOTTOM_F + CONSOLE_H_F + GAP_ABOVE_CONSOLE_F,
        hspace=0.35, wspace=0.25
    )

    #Create console
    console_ax = fig.add_axes([0, 0, 1, 1])
    console_ax.set_facecolor("black")
    console_ax.set_xticks([]);
    console_ax.set_yticks([])
    for s in console_ax.spines.values():
        s.set_color("red");
        s.set_linewidth(1.5)

    #Text obs
    _instr_text = None
    _log_text = None
    _rule_line = None
    _clip_rect = None

    #Mirror console
    _buf = deque()

    #All functions are intentionally clearly named so comments arent required

    def _fig_px():
        w_in, h_in = fig.get_size_inches()
        return int(round(w_in * fig.dpi)), int(round(h_in * fig.dpi))

    def _console_bounds_fig():
        fig_w_px, fig_h_px = _fig_px()
        width_px = max(120, fig_w_px - 30)
        left_px = (fig_w_px - width_px) / 2.0
        #Convert the px to fractions
        left_f = left_px / fig_w_px
        width_f = width_px / fig_w_px
        return [left_f, BOTTOM_F, width_f, CONSOLE_H_F]

    def _visible_log_lines():
        bbox = console_ax.get_window_extent()
        h_px = max(1, int(bbox.height))
        avail_px = _rule_y_px() - PAD_BOT_PX
        avail_px = max(0, int(avail_px))
        return max(1, avail_px // max(1, LINE_PX))

    def _rule_y_px():
        bbox = console_ax.get_window_extent()
        h_px = max(1, int(bbox.height))
        y_rule_from_top_px = PAD_TOP_PX + INSTR_HEIGHT_PX + RULE_GAP_PX
        return max(0, h_px - y_rule_from_top_px)

    def _rule_y_axes():
        bbox = console_ax.get_window_extent()
        return max(0.0, min(1.0, _rule_y_px() / max(1, int(bbox.height))))

    def _trim_to_visible():
        max_lines = _visible_log_lines()
        while len(_buf) > max_lines:
            _buf.popleft()

    def _render_console(force=False):
        nonlocal _instr_text, _log_text, _rule_line, _clip_rect

        instr_str = "\n".join(INSTR_LINES)
        tr_instr = console_ax.transAxes + _mtransforms.ScaledTranslation(
            PAD_LEFT_PX / fig.dpi, -PAD_TOP_PX / fig.dpi, fig.dpi_scale_trans
        )
        tr_log = console_ax.transAxes + _mtransforms.ScaledTranslation(
            PAD_LEFT_PX / fig.dpi, PAD_BOT_PX / fig.dpi, fig.dpi_scale_trans
        )

        if _instr_text is None:
            _instr_text = console_ax.text(
                0.0, 1.0, instr_str, ha="left", va="top",
                color="white", family="monospace", fontsize=CONSOLE_FONT,
                transform=tr_instr
            )
        else:
            _instr_text.set_transform(tr_instr)
            _instr_text.set_text(instr_str)

        y_rule = _rule_y_axes()
        if _rule_line is None:
            _rule_line = Line2D([0, 1], [y_rule, y_rule], color="red", linewidth=2, transform=console_ax.transAxes)
            console_ax.add_line(_rule_line)
        else:
            _rule_line.set_ydata([y_rule, y_rule])

        if _clip_rect is None:
            _clip_rect = Rectangle(
                (0, 0), 1, y_rule, transform=console_ax.transAxes,
                facecolor="none", edgecolor="none"
            )
            console_ax.add_patch(_clip_rect)
        else:
            _clip_rect.set_height(y_rule)

        _trim_to_visible()
        tail = list(_buf)
        log_str = "\n".join(tail)

        if _log_text is None:
            _log_text = console_ax.text(
                0.0, 0.0, log_str, ha="left", va="bottom",
                color="white", family="monospace", fontsize=CONSOLE_FONT,
                transform=tr_log, clip_on=True
            )
            _log_text.set_clip_path(_clip_rect)
        else:
            _log_text.set_transform(tr_log)
            _log_text.set_text(log_str)
            _log_text.set_clip_path(_clip_rect)

        fig.canvas.draw_idle()

    def _reflow(_evt=None):
        console_ax.set_position(_console_bounds_fig())
        _render_console(force=True)

    #Initial position & draw
    _reflow()
    fig.canvas.mpl_connect("resize_event", _reflow)

    #Auto scroll
    if not getattr(sys.stdout, "_is_console_mirror", False):
        _orig_stdout = sys.stdout

        class _MirrorStdout:
            _is_console_mirror = True

            def __init__(self, wrapped, skip_first_lines: int = 3):
                self._wrapped = wrapped
                self._acc = ""
                self._skip = skip_first_lines

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

            def write(self, s):
                self._wrapped.write(s)
                self._acc += s
                while "\n" in self._acc:
                    line, self._acc = self._acc.split("\n", 1)
                    if self._skip > 0:
                        self._skip -= 1
                    else:
                        _buf.append(line)
                _trim_to_visible()
                _render_console()

            def flush(self):
                try:
                    self._wrapped.flush()
                except Exception:
                    pass

        sys.stdout = _MirrorStdout(_orig_stdout)

    #Finally end of my console and graph window block

    try:
        fig.canvas.manager.set_window_title("Astro Alley - PPO Training Dashboard")
    except Exception:
        pass

    #Lightweight to prevent the lagging i was having
    def ui_poll():
        if not plt.fignum_exists(fig.number):
            return
        fig.canvas.flush_events()
        plt.pause(.001)

    #Color segments based on trend up or down
    def make_line(ax):
        lc = LineCollection([], linewidths=2.0)
        ax.add_collection(lc)
        #Stay grey no change
        sca = ax.scatter([],[], s=20, alpha=0.9)

        return lc, sca

    lc_tlen, sc_tlen,  = make_line(ax_tlen)
    lc_tret, sc_tret,  = make_line(ax_tret)
    lc_eval_len, sc_eval_len = make_line(ax_eval_len)
    lc_eval_pipes, sc_eval_pipes = make_line(ax_eval_pipes)

    #Update my line collection
    def set_slope_colored_segments(lc: LineCollection, scattr, xs, ys):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if len(xs) == 0:
            lc.set_segments([])
            scattr.set_offsets(np.empty((0, 2)))
            return

        if len(xs) == 1:
            lc.set_segments([])
            pts = np.column_stack([xs, ys])
            scattr.set_offsets(pts)
            scattr.set_facecolor(["grey"])

            #Update axis
            ax= scattr.axes
            ax.update_datalim(pts)
            ax.autoscale_view()
            return

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc.set_segments(segments)

        seg_colors = []
        dot_colors = ["gray"]

        for i in range(1, len(ys)):
            if ys[i] > ys[i-1]:
                c = "green"
            elif ys[i] < ys[i-1]:
                c = "red"
            else:
                c = "gray"

            seg_colors.append(c)
            dot_colors.append(c)

        lc.set_color(seg_colors)

        pts = np.column_stack([xs, ys])
        scattr.set_offsets(pts)
        scattr.set_facecolor(dot_colors)

        ax = scattr.axes
        ax.update_datalim(pts)
        ax.autoscale_view()

        #Update per color
        scattr.set_offsets(np.column_stack([xs, ys]))
        scattr.set_facecolor(dot_colors)

    def update_plots(current_update: int, elapsed_sec: float):
        if not plt.fignum_exists(fig.number):
            return

        elapsed_str = format_elapsed(elapsed_sec)
        fig.suptitle(
            f"Astro Alley - Update #{current_update} | Elapsed {elapsed_str}  (Press 'Q' to stop training)",
            color="white"
        )

        #Top row
        set_slope_colored_segments(lc_tlen, sc_tlen, time_hist, train_len_hist)
        set_slope_colored_segments(lc_tret, sc_tret, time_hist, train_ret_hist)

        #Bottom row
        set_slope_colored_segments(lc_eval_len, sc_eval_len, updates_hist, eval_len_hist)
        set_slope_colored_segments(lc_eval_pipes, sc_eval_pipes, updates_hist, eval_pipes_hist)

        #Rescale with our new data
        for ax in (ax_tlen, ax_tret, ax_eval_len, ax_eval_pipes):
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw()
        ui_poll()

    is_fullscreen = False

    def _toggle_fullscreen():
        nonlocal is_fullscreen
        try:
            manager = plt.get_current_fig_manager()
        except Exception:
            return

        window = getattr(manager, "window", None)
        if window is not None and hasattr(window, "showFullScreen") and hasattr(window, "showNormal"):
            if is_fullscreen:
                window.showNormal()
                is_fullscreen = False
            else:
                window.showFullScreen()
                is_fullscreen = True
        elif hasattr(manager, "full_screen_toggle"):
            manager.full_screen_toggle()
            is_fullscreen = not is_fullscreen

    def _on_key(event):
        #ESC exits fullscreen back to windowed, but doesnt stop my training
        if event.key == "escape" and is_fullscreen:
            _toggle_fullscreen()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    #Bring my graph window to the front on open
    plt.show(block=False)
    ui_poll()
    #Immediately go fullscreen
    _toggle_fullscreen()

    #Dont steal focus after first time
    try:
        import matplotlib as mpl
        mpl.rcParams["figure.raise_window"] = False
    except Exception:
        pass


    #Defaults for my PPO setup
    total_updates=100000
    steps_per_roll = 8192
    mini_batch_size = 1024
    ppo_epochs = 6

    initial_lr = 3e-4 #lr = learning rate, throughout my class
    final_lr = 3e-5

    clip_eps = 0.15
    vf_coef = 0.5
    entropy_coef_start = 0.012
    entropy_coef_end = .003
    target_kl= .02 #kl= divergence for early stop, also throughout class
    max_grad_norm = 0.7

    opt = optim.Adam(model.parameters(), lr=initial_lr, eps=1e-5) #Adam optimizer for model parameters

    best_eval_pipes = -1.0 #Track best eval
    save_path= os.path.join(os.path.dirname(__file__),"autopilot_policy.pt") #Save best model
    best_state_dict = None #Keep track best model

    print("****** Stop the program at anytime and the best policy so far will be saved and training will end. ******")
    print("****** Note: Each update = 50 Episodes. ******")
    print("****** Press 'Q' to stop training and save best model. Note: Focus must be in the graph window******")

    #History for graph
    updates_hist: list[int] = []
    time_hist: list[float] = []
    train_len_hist: list[float] = []
    train_ret_hist: list[float] = []
    eval_len_hist: list[float] = []
    eval_pipes_hist: list[float] = []

    last_eval_len = 0.0
    last_eval_pipes = 0.0

    EVAL_INTERVAL = 10
    EVAL_EPISODES = 50
    MAX_POINTS = 1000

    start_time = time.time()

    try:
        #PPO outer loop over updates
        for update in range(1,total_updates+1):
            #If graphs window closed end program
            if not plt.fignum_exists(fig.number):
                print("\nPlot window closed. Training loop stopped.")
                break

            #Check for q
            if HAVE_MSVCRT and msvcrt.kbhit():
                ch=msvcrt.getwch()
                if ch.lower() == 'q':
                    print("\n 'Q' pressed. Training loop stopped, saving best model...")
                    raise KeyboardInterrupt

            ui_poll()

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
                for step in range(steps_per_roll):
                    #Check for 'Q'
                    if HAVE_MSVCRT and msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch.lower() == 'q':
                            print("\n'q' pressed. Stopping training and saving best model...")
                            raise KeyboardInterrupt

                    #Check UI poll so drag is not laggy
                    if step % 256 == 0:
                        ui_poll()

                    with torch.inference_mode():
                        x= torch.tensor(obs, dtype= torch.float32, device=("cpu" if INFER_ON_CPU else device)).unsqueeze(0)
                        logits, value = model_rollout(x) #Pass forward
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
            obs_np = np.asarray(obs_buf, dtype=np.float32)
            obs_np = np.ascontiguousarray(obs_np, dtype=np.float32)
            obs_t = torch.from_numpy(obs_np).to(device)

            act_t =torch.as_tensor(act_buf, dtype=torch.int64, device=device)
            old_logp_t= torch.as_tensor(logp_buf, dtype=torch.float32, device=device)
            reward_t= torch.as_tensor(reward_buf, dtype=torch.float32, device=device)
            done_t= torch.as_tensor(done_buf, dtype=torch.float32, device=device)
            val_t =torch.as_tensor(val_buf, dtype=torch.float32, device=device)

            #Compute GAE returns and advantages
            adv_t, ret_t = gae_return(reward_t, val_t, done_t, gamma=.99, lam=.95, last_val=last_val,)
            adv_t = (adv_t- adv_t.mean()) / (adv_t.std() + 1e-8) #Normalize

            #Update PPO
            for _ in range(ppo_epochs):
                kl_exceeded = False

                #Loop over mini batches
                mb_counter=0

                for (b_obs, b_act, b_old_logp, b_adv, b_ret, b_val_old,) in batchify_torch(
                        mini_batch_size, obs_t,act_t, old_logp_t,adv_t, ret_t,val_t): #Draw mini batches

                    mb_counter+=1

                    #One more check in opt
                    if HAVE_MSVCRT and msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch.lower() == 'q':
                            print("\n'q' pressed. Stopping training and saving best model...")
                            raise KeyboardInterrupt

                    if mb_counter % 4 == 0:
                        ui_poll()

                    #Convert mini batch nack to tensor on device
                    #b_obs = torch.tensor(b_obs, dtype=torch.float32, device=device)
                    #b_act = torch.tensor(b_act, dtype=torch.int64, device=device)
                    #b_old_logp = torch.tensor(b_old_logp, dtype=torch.float32, device=device)
                    #b_adv = torch.tensor(b_adv, dtype=torch.float32, device=device)
                    #b_ret = torch.tensor(b_ret, dtype=torch.float32, device=device)
                    #b_val_old = torch.tensor(b_val_old, dtype=torch.float32, device=device)

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

            if INFER_ON_CPU:
                model_rollout.load_state_dict(model.state_dict())
                model_rollout.eval()

            avg_len = float(np.mean(ep_lens)) if ep_lens else 0.0 #Average episode len across rollout
            avg_ret = float(np.mean(ep_rets)) if ep_rets else 0.0 #Average episode return across rollout

            #Display statistics per update and per 10 updates
            do_eval = (update % EVAL_INTERVAL == 0)
            if do_eval:
                eval_len, eval_pipes = evaluate_policy(model, device, episodes=EVAL_EPISODES) #Episodes is per update
                last_eval_len = eval_len
                last_eval_pipes = eval_pipes

                if eval_pipes > best_eval_pipes: #If new best agent
                    best_eval_pipes = eval_pipes
                    best_state_dict = copy.deepcopy(model.state_dict()) #Copy best parameters
                    torch.save(best_state_dict, save_path) #Save best so far

            now = time.time()
            elapsed = now - start_time
            elapsed_minutes = elapsed / 60

            #My log history
            updates_hist.append(update)
            time_hist.append(elapsed_minutes)
            train_len_hist.append(avg_len)
            train_ret_hist.append(avg_ret)
            eval_len_hist.append(last_eval_len)
            eval_pipes_hist.append(last_eval_pipes)

            #Keep only last max points to keep plotting fast
            if len(updates_hist)> MAX_POINTS:
                updates_hist = updates_hist[-MAX_POINTS:]
                time_hist = time_hist[-MAX_POINTS:]
                train_ret_hist = train_ret_hist[-MAX_POINTS:]
                train_len_hist = train_len_hist[-MAX_POINTS:]
                eval_pipes_hist = eval_pipes_hist[-MAX_POINTS:]
                eval_len_hist = eval_len_hist[-MAX_POINTS:]

            #Console log
            if do_eval:
                print(f"[Update #: {update:05d} | Elapsed: {format_elapsed(elapsed)}] "
                      f"train_length = {avg_len:6.1f} | train_return = {avg_ret:+.2f} "
                      f"| ent = {ent_coef:.4f} | learning_rate_now = {learning_rate_now:.6f} "
                      f"| eval_length = {last_eval_len:6.1f} | eval_pipes = {last_eval_pipes:5.2f} ")
            else:
                print(f"[Update #: {update:05d} | Elapsed: {format_elapsed(elapsed)}] "
                      f"train_length = {avg_len:6.1f} | train_return = {avg_ret:+.2f} "
                      f"| ent = {ent_coef:.4f} | learning_rate_now = {learning_rate_now:.6f} ")

            #Update UI every ppo update
            update_plots(update, elapsed)

    #Quick escape
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt / 'Q' received. Saving the best model so far...")
        if best_state_dict is not None: #Pick best model if we have one
            torch.save(best_state_dict, save_path) #Save best weights
            print("Best model (by eval_pipes) saved: ", save_path)
        else:
            torch.save(model.state_dict(), save_path)
            print("No eval yet, current model saved: ", save_path)

        #Final refresh in loop
        if updates_hist:
            final_elapsed = time.time() - start_time
            update_plots(updates_hist[-1], final_elapsed)

        plt.ioff()
        plt.show()
        return

    #Refresh after loop
    if updates_hist:
        final_elapsed = time.time() - start_time
        update_plots(updates_hist[-1], final_elapsed)

    if not os.path.isfile(save_path): #Make sure something is saved
        to_save = best_state_dict if best_state_dict is not None else model.state_dict()
        torch.save(to_save, save_path)
        print("$#$#$#$ Check Save! $#$#$#$")

    #Final
    print("Final best_eval_pipes:", best_eval_pipes)
    print("Saved:", save_path)

    #Keep final plot window open
    plt.ioff()
    plt.show()

#Run it
if __name__ == "__main__":
    print(f"\n******ARE YOU SURE YOU WANT TO START A NEW TRAINING SESSION???******\n"
          "******DOING SO WILL OVERWRITE THE PREVIOUS SAVED AUTOPILOT_POLICY.pt******\n")
    answer= input("Do you want to continue? (y/n): ")

    if answer not in ("Yes", "yes", "y", "Y"):
        print("Training Aborted Before Start!")
        sys.exit(0)

    train()