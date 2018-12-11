#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

paths = glob("./*/log.txt")
for path in paths:
	with open(path) as f:
	    lines = f.readlines()
	    frames = []
	    rewards = []
	    for line in lines:
	    	frame = float(line.split(',')[2].strip().split()[1][:-1])
	    	reward = float(line.split(',')[3].strip().split()[2])
	    	frames.append(frame)
	    	rewards.append(reward)
	    frames1 = np.arange(frames[0], frames[-1], 0.1)
	    frames1 = np.around(frames1, decimals=1)
	    rewards1 = []
	    frames2 = []
	    j = 0
	    for i in range(len(frames1)):
	    	reward = 0
	    	counter = 0
	    	while frames[j] == frames1[i]:
	    		reward += rewards[j]
	    		counter += 1
	    		j += 1
	    	if counter > 0:
	    		frames2.append(frames1[i])
	    		rewards1.append(float(reward)/counter)
	    title = path.split('/')[1]
	    plt.title(title)
	    plt.plot(frames2, rewards1)
	    plt.xlabel('Time steps (in millions)')
	    plt.ylabel('Rewards')
	    plt.savefig('./'+title+'/plot.png')
	    plt.close()