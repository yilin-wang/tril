import argparse
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
import pygame
import sys
import time
import matplotlib
import numpy as np
import pickle

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
from baselines.common.trex_utils import preprocess

sys.path[0] += '/baselines'
from baselines.common.trex_utils import preprocess
# from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except Exception:
    pass

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)



    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj.permute(0,3,1,2) #get into NCHW format
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        # x = x.view(-1, 784)
        x = x.reshape(-1,784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j

def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            r = 1/(1+np.exp(-r))
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))

if __name__ == '__main__':
    num_trajs = 2000
    num_snippets = 6000
    num_super_snippets = 0
    min_snippet_length = 50 #length of trajectory for training comparison
    max_snippet_length = 100

    lr = 0.00005
    weight_decay = 0.0
    num_iter = 5 #num times through training data
    l1_reg = 0.0
    stochastic = True

    demonstrations = {}
    for i in range(12):
        with open('col1_demos/%d' % (i+1),'rb') as fp:
            dem = pickle.load(fp)
        demonstrations[i] = dem

    # human_rankings = []
    # label_reader = open("human_labels/si_columns.csv")
    # for i,line in enumerate(label_reader):
    #     if i == 0:
    #         continue #skip header info
    #     parsed = line.split(",")
    #     a_index = int(parsed[0])
    #     b_index = int(parsed[1])
    #     label = int(parsed[2])
    #     human_rankings.append((a_index, b_index, label))

    reward_model_path = './learned_models/col1.params'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.load_state_dict(torch.load(reward_model_path))
    reward_net.to(device)
  
    with torch.no_grad():
        pred_returns = [(predict_traj_return(reward_net, traj), len(traj)) for traj in demonstrations.values()]
    for i, p in enumerate(pred_returns):
        print(i+1,p[0],p[1],p[0]/p[1])