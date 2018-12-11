from __future__ import print_function

import os, sys, time, argparse 
import torch, gym, glob
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from scipy.signal import lfilter

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1]

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v0', type=str, help='gym environment')
    parser.add_argument('--trained_env', default='Pong-v0', type=str, help='trained on gym environment')
    parser.add_argument('--processes', default=20, type=int, help='training processes')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--test', default=False, type=bool, help='play according to saved model')
    parser.add_argument('--steps', default=20, type=int, help='forward steps')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount (GAE)')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    return parser.parse_args()

def printlog(dataPath, s, mode='a'):
    print(s)
    f=open(dataPath+'log_test.txt',mode)
    f.write(s+'\n')
    f.close()

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = -(-logps * torch.exp(logps)).sum()
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'] = torch.zeros(1)
                state['shared_steps'].share_memory_()
                state['step'] = 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'].share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1
            super.step(closure)
