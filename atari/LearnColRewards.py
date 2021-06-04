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

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
parser.add_argument('--reward_net_path', default='', help="name and location for learned model params")
parser.add_argument('--seed', default=0, help="random seed for experiments")
parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
parser.add_argument('--save_fig_dir', help ="where to save visualizations")

# def create_training_data(demonstrations, human_rankings, num_trajs, num_snippets, min_snippet_length, max_snippet_length):
#     #collect training data
#     max_traj_length = 0
#     training_obs = []
#     training_labels = []
#     num_demos = 12

#     print(human_rankings)

#     #add full trajs (for testing on enduro)
#     for n in range(num_trajs):
#         random_pref = random.choice(human_rankings)
#         #pick random pair from mturk rankings
#         ti,tj,label = random_pref

#         #create random partial trajs by finding random start frame and random skip frame
#         si = np.random.randint(6)
#         sj = np.random.randint(6)
#         step = np.random.randint(3,7)
#         #step_j = np.random.randint(2,6)
#         #print("si,sj,skip",si,sj,step)
#         traj_i = demonstrations[ti][si::step]  #slice(start,stop,step)
#         traj_j = demonstrations[tj][sj::step]
#         #max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
#         #print("label", label)

#         training_obs.append((traj_i, traj_j))
#         training_labels.append(label)
#         max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

#     #random size snippets with progress prior
#     for n in range(num_snippets):

#         random_pref = random.choice(human_rankings)
#         #pick random pair from Mturk rankings
#         ti,tj,label = random_pref
#         #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
#         min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
#         rand_length = np.random.randint(min_snippet_length, max_snippet_length)
#         if label == 1: #pick tj snippet to be later than ti
#             ti_start = np.random.randint(min_length - rand_length + 1)
#             tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
#         else: #ti is better so pick later snippet in ti
#             tj_start = np.random.randint(min_length - rand_length + 1)
#             #print(tj_start, len(demonstrations[ti]))
#             ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
#         #print("start", ti_start, tj_start)
#         traj_i = demonstrations[ti][ti_start:ti_start+rand_length:2] #skip everyother framestack to reduce size
#         traj_j = demonstrations[tj][tj_start:tj_start+rand_length:2]
#         max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
#         training_obs.append((traj_i, traj_j))
#         training_labels.append(label)

#     print("maximum traj length", max_traj_length)
#     return training_obs, training_labels

def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)

    #add full trajs (for use on Enduro)
    for n in range(num_trajs):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random partial trajs by finding random start frame and random skip frame
        si = np.random.randint(6)
        sj = np.random.randint(6)
        step = np.random.randint(3,7)
        
        traj_i = demonstrations[ti][si::step]  #slice(start,stop,step)
        traj_j = demonstrations[tj][sj::step]
        
        if ti > tj:
            label = 0
        else:
            label = 1
        
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))


    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj: #pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
        traj_i = demonstrations[ti][ti_start:ti_start+rand_length:2] #skip everyother framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start+rand_length:2]

        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels

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

# Now we train the network.
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 500 == 499:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")





def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)






def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
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

    # training_obs, training_labels = create_training_data(demonstrations, human_rankings, num_trajs, num_snippets, min_snippet_length, max_snippet_length)
    training_obs, training_labels = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    # reward_model_path = './learned_models/col1.params'
    reward_model_path = args.reward_net_path
    learn_reward(reward_net, optimizer, training_obs, training_labels, num_iter, l1_reg, reward_model_path)

    torch.save(reward_net.state_dict(), reward_model_path)
  
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations.values()]
    for i, p in enumerate(pred_returns):
        print(i+1,p)

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))
