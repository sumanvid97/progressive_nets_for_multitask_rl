from __future__ import print_function

import os, sys, time, argparse 
import torch, gym, glob
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from scipy.misc import imresize

from utils import SharedAdam, get_args, cost_func, printlog
from a3c import ActorCritic

os.environ['OMP_NUM_THREADS'] = '1'
prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

class ConvLayer(nn.Module):
    def __init__(self, col, depth, n_in, n_out):
        super(ConvLayer, self).__init__()
        self.col = col
        self.depth = depth
        self.n_in = n_in
        self.n_out = n_out
        self.layer = nn.Conv2d(n_in, n_out, 3, stride=2, padding=1)
        if col > 0:
            self.u = nn.Conv2d(n_in, n_out, 3, stride=2, padding=1)

    def forward(self, inputs):
        if self.col == 1:
            current_out = self.layer(inputs[1])
            prev_out = self.u(inputs[0])
            return F.elu(current_out + prev_out)
        else:
            current_out = self.layer(inputs)
            return F.elu(current_out)
    
class GruLayer(nn.Module):
    def __init__(self, col, depth):
        super(GruLayer, self).__init__()
        self.col = col
        self.depth = depth
        self.layer = nn.GRUCell(32 * 5 * 5, 256)
        if col > 0:
            self.u = nn.GRUCell(32 * 5 * 5, 256)

    def forward(self, inputs, hx):
        current_out = self.layer(inputs, hx)
        return current_out

class LinearLayer(nn.Module):
    def __init__(self, col, depth, n_out):
        super(LinearLayer, self).__init__()
        self.col = col
        self.depth = depth
        self.layer = nn.Linear(256, n_out)
        if col > 0:
            self.u = nn.Linear(256, n_out)

    def forward(self, inputs):
        if self.col == 1:
            current_out = self.layer(inputs[1])
            prev_out = self.u(inputs[0])
            return current_out + prev_out
        else:
            current_out = self.layer(inputs)
            return current_out

# a progressive neural network        
class PNN(nn.Module):
    def __init__(self, num_actions):
        super(PNN, self).__init__()
        self.num_actions = num_actions
        self.columns = nn.ModuleList([])
        self.col = -1

    def forward(self, inpts):
        inpts, hx = inpts
        inpts = (inpts, inpts)
        inputs = (self.columns[0][0](inpts[0]), self.columns[1][0](inpts))
        for depth in range(1, 4):
            inputs = (self.columns[0][depth](inputs[0]), self.columns[1][depth](inputs))
        inputs = inputs[0].view(-1, 32 * 5 * 5), inputs[1].view(-1, 32 * 5 * 5)
        hx1 = self.columns[0][4](inputs[0], (hx))
        hx2 = self.columns[1][4](inputs[1], (hx))
        critic_linear = self.columns[1][5]((hx1, hx2))
        actor_linear = self.columns[1][6]((hx1, hx2))
        return critic_linear, actor_linear, hx2

    def new_task(self):
        col = len(self.columns)
        modules = [ConvLayer(col, 0, 1, 32)]
        for depth in range(1,4):
            modules.append(ConvLayer(col, depth, 32, 32))
        modules.append(GruLayer(col, 4))
        modules.append(LinearLayer(col, 5, 1))
        modules.append(LinearLayer(col, 6, self.num_actions))
        new_column = nn.ModuleList(modules)
        self.columns.append(new_column)

    def load(self, model):
        self.columns[0][0] = model.conv1
        self.columns[0][1] = model.conv2
        self.columns[0][2] = model.conv3
        self.columns[0][3] = model.conv4
        self.columns[0][4] = model.gru
        self.columns[0][5] = model.critic_linear
        self.columns[0][6] = model.actor_linear

    def freeze_column(self):
        for params in self.columns[0].parameters():
            params.requires_grad = False

    def parameters(self):
        return self.columns[1].parameters()

def train(shared_model, shared_optimizer, weights, rank, args, info):
    env = gym.make(args.env)
    env.seed(args.seed + rank)
    
    torch.manual_seed(args.seed + rank)
    model = PNN(num_actions=args.num_actions)
    model.new_task()
    model.load(weights)
    model.freeze_column()
    model.new_task()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    state = torch.tensor(prepro(env.reset()))

    start_time = last_disp_time = time.time()
    episode_length = 0
    episode_reward = 0
    episode_loss = 0
    done = True

    while info['frames'][0] <= 8e7 or args.test:
        model.load_state_dict(shared_model.state_dict())

        if done: 
            hx = torch.zeros(1, 256)
        else:
            hx = hx.detach()
        values = []
        logps = []
        actions = []
        rewards = []

        for step in range(args.steps):
            episode_length += 1
            value, logit, hx = model((state.view(1,1,80,80), hx))
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]
            state, reward, done, _ = env.step(action.numpy()[0])
            # if args.test:
            #     env.render()

            state = torch.tensor(prepro(state))
            episode_reward += reward
            reward = np.clip(reward, -1, 1)
            done = done or episode_length >= 1e4
            
            info['frames'].add_(1)
            num_frames = int(info['frames'].item())
            if num_frames % 4e6 == 0:
                torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

            if done:
                info['episodes'] += 1
                if info['episodes'][0] == 1:
                    interp = 1
                else:
                    interp = 1 - args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * episode_reward)
                info['run_loss'].mul_(1-interp).add_(interp * episode_loss)

            if rank == 0 and time.time() - last_disp_time > 60:
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args.save_dir, 'time {}, episodes {:.0f}, frames {:.1f}M, mean episode_reward {:.2f}, run loss {:.2f}'
                    .format(elapsed, info['episodes'].item(), num_frames/1e6,
                    info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done:
                episode_length, episode_reward, episode_loss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

            values.append(value)
            logps.append(logp)
            actions.append(action)
            rewards.append(reward)

        if done:
            next_value = torch.zeros(1,1)
        else:
            next_value = model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        episode_loss += loss.item()
        shared_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        shared_optimizer.step()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = get_args()
    args.save_dir = 'pnn-{}/'.format(args.env.lower())
    if args.test:
        args.processes = 1
        args.lr = 0
    args.num_actions = gym.make(args.env).action_space.n
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    weights = ActorCritic(gym.make(args.trained_env).action_space.n)
    step = weights.load('{}trained/model.8.tar'.format(args.save_dir))
    torch.manual_seed(args.seed)
    shared_model = PNN(num_actions=args.num_actions).share_memory()
    shared_model.new_task()
    shared_model.load(weights)
    shared_model.freeze_column()
    shared_model.new_task()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += 8 * 1e6
    if int(info['frames'].item()) == 0:
        printlog(args.save_dir,'', mode='w')
    
    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, weights, rank, args, info))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()