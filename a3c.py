from __future__ import print_function

import os, sys, time, argparse 
import torch, gym, glob
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from scipy.misc import imresize

from utils import SharedAdam, get_args, cost_func, printlog

os.environ['OMP_NUM_THREADS'] = '1'
prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

# an actor-critic neural network
class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 5 * 5)
        hx = self.gru(x, (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def load(self, model):
        paths = glob.glob(model)
        step = 0
        if len(paths) != 0:
            step = int(paths[0].split('.')[-2])
            self.load_state_dict(torch.load(paths[0]))
        if step==0:
            print(model + " is not a saved models")
        else:
            print("loaded model: {}".format(paths[0]))
        return step

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar')
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts)
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        if step==0:
            print("no saved models")
        else:
            print("loaded model: {}".format(paths[ix]))
        return step

def train(shared_model, shared_optimizer, rank, args, info):
    env = gym.make(args.env)
    env.seed(args.seed + rank)
    
    torch.manual_seed(args.seed + rank)
    model = ActorCritic(num_actions=args.num_actions)
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
    args.save_dir = 'a3c-{}/'.format(args.env.lower())
    if args.test:
        args.processes = 1
        args.lr = 0
    args.num_actions = gym.make(args.env).action_space.n
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    shared_model = ActorCritic(num_actions=args.num_actions).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0:
        printlog(args.save_dir,'', mode='w')
    
    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
