import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
import matplotlib.pylab as plt
import argparse
import pickle

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
parser.add_argument('--reward_net_path', default='', help="name and location for learned model params")
parser.add_argument('--seed', default=0, help="random seed for experiments")
parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
parser.add_argument('--save_fig_dir', help ="where to save visualizations")
parser.add_argument('--demo_dir', help ="where to find demos")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)


args = parser.parse_args()
env_name = args.env_name
save_fig_dir = args.save_fig_dir

# if env_name == "spaceinvaders":
#     env_id = "SpaceInvadersNoFrameskip-v4"
# elif env_name == "mspacman":
#     env_id = "MsPacmanNoFrameskip-v4"
# elif env_name == "videopinball":
#     env_id = "VideoPinballNoFrameskip-v4"
# elif env_name == "beamrider":
#     env_id = "BeamRiderNoFrameskip-v4"
# else:
#     env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"
env_type = "atari"

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# print(env_id)

stochastic = True

reward_net_path = args.reward_net_path


env = make_vec_env("SpaceInvadersNoFrameskip-v4", 'atari', 1, 0,
                   wrapper_kwargs={
                       'clip_rewards':False,
                       'episode_life':False,
                   })


env = VecFrameStack(env, 4)
agent = PPO2Agent(env, env_type, stochastic)



import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        #self.fc1 = nn.Linear(1936,64)
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
        return torch.cat([cum_r_i, cum_r_j]), abs_r_i + abs_r_j

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


reward = Net()
reward.load_state_dict(torch.load(reward_net_path)) # perhaps change for gpu
reward.to(device)

from baselines.common.trex_utils import preprocess
# model_dir = args.models_dir
demo_dir = args.demo_dir

demonstrations = []
for i in range(12):
    # with open('row1_demos/%d' % (i+1),'rb') as fp:
    with open(demo_dir + '/%d' % (i+1),'rb') as fp:
        dem = pickle.load(fp)
    demonstrations.append(dem)

indices = []
min_reward = 100000
max_reward = -100000
cnt = 0
with torch.no_grad():
    for j,d in enumerate(demonstrations):
        print(cnt)
        cnt += 1
        for i,s in enumerate(d[2:-1]):
            r = reward.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            indices.append([j,i,r])
            if r < min_reward:
                min_reward = r
                min_frame = s
                min_frame_i = i+2
            elif r > max_reward:
                max_reward = r
                max_frame = s
                max_frame_i = i+2

rewards = np.array([ind[2] for ind in indices])
ninety = np.percentile(rewards, 95, interpolation='nearest')
nind, = np.where(np.isclose(rewards,ninety))
nind = int(nind[0])
eighty = np.percentile(rewards, 90, interpolation='nearest')
eind, = np.where(np.isclose(rewards,eighty))
eind = int(eind[0])

j = indices[nind][0]
i = indices[nind][1]

frame_95 = demonstrations[j][i+2]

j = indices[eind][0]
i = indices[eind][1]

frame_90 = demonstrations[j][i+2]

def mask_coord(i,j,frames, mask_size, channel):
    #takes in i,j pixel and stacked frames to mask
    masked = frames.copy()
    # masked[:,i:i+mask_size,j:j+mask_size,channel] = 0
    masked[i:i+mask_size,j:j+mask_size,channel] = 0
    return masked

def gen_attention_maps(frames, mask_size):

    orig_frame = frames

    #okay so I want to vizualize what makes these better or worse.
    # _,height,width,channels = orig_frame.shape
    height,width,channels = orig_frame.shape

    #find reward without any masking once
    r_before = reward.cum_return(torch.from_numpy(np.array([orig_frame])).float().to(device))[0].item()
    heat_maps = []
    for c in range(4): #four stacked frame channels
        delta_heat = np.zeros((height, width))
        for i in range(height-mask_size):
            for j in range(width - mask_size):
                #get masked frames
                masked_ij = mask_coord(i,j,orig_frame, mask_size, c)
                r_after = r = reward.cum_return(torch.from_numpy(np.array([masked_ij])).float().to(device))[0].item()
                r_delta = abs(r_after - r_before)
                #save to heatmap
                delta_heat[i:i+mask_size, j:j+mask_size] += r_delta
        heat_maps.append(delta_heat)
    return heat_maps



#plot heatmap
mask_size = 3
delta_heat_max = gen_attention_maps(max_frame, mask_size)
delta_heat_min = gen_attention_maps(min_frame, mask_size)
delta_heat_95 = gen_attention_maps(frame_95, mask_size)
delta_heat_90 = gen_attention_maps(frame_90, mask_size)

plt.figure(1)
# for cnt in range(4):
#     plt.subplot(1,4,cnt+1)
#     plt.imshow(delta_heat_95[cnt],cmap='seismic', interpolation='nearest')
#     plt.axis('off')
plt.imshow(delta_heat_95[1],cmap='seismic', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "95_attention.png", bbox_inches='tight')


plt.figure(2)
# for cnt in range(4):
#     plt.subplot(1,4,cnt+1)
#     plt.imshow(delta_heat_95[cnt],cmap='seismic', interpolation='nearest')
#     plt.axis('off')
plt.imshow(frame_95[:,:,1])
plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "95_frame.png", bbox_inches='tight')

plt.figure(3)
# for cnt in range(4):
#     plt.subplot(1,4,cnt+1)
#     plt.imshow(delta_heat_95[cnt],cmap='seismic', interpolation='nearest')
#     plt.axis('off')
plt.imshow(delta_heat_90[1],cmap='seismic', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "90_attention.png", bbox_inches='tight')

plt.figure(4)
# for cnt in range(4):
#     plt.subplot(1,4,cnt+1)
#     plt.imshow(delta_heat_95[cnt],cmap='seismic', interpolation='nearest')
#     plt.axis('off')
plt.imshow(frame_90[:,:,1])
plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "90_frame.png", bbox_inches='tight')

# In[45]:


plt.figure(5)
# for cnt in range(4):
#     plt.subplot(1,4,cnt+1)
#     plt.imshow(delta_heat_max[cnt],cmap='seismic', interpolation='nearest')
#     plt.axis('off')
plt.imshow(delta_heat_max[1],cmap='seismic', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "max_attention.png", bbox_inches='tight')


plt.figure(6)
# print(max_frame_i)
# print(max_reward)
# for cnt in range(4):
#     plt.subplot(1,4,cnt+1)
#     plt.imshow(max_frame[:,:,cnt])
#     plt.axis('off')
# plt.imshow(max_frame)
# plt.axis('off')
plt.imshow(max_frame[:,:,1])
plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "max_frame.png", bbox_inches='tight')


plt.figure(7)
# for cnt in range(4):
#     plt.subplot(1,4,cnt+1)
#     plt.imshow(delta_heat_min[cnt],cmap='seismic', interpolation='nearest')
#     plt.axis('off')
plt.imshow(delta_heat_min[1],cmap='seismic', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "min_attention.png", bbox_inches='tight')

plt.figure(8)
# for cnt in range(4):
#     plt.subplot(1,4,cnt+1)
#     plt.imshow(min_frame[:,:,cnt])
#     plt.axis('off')
# plt.imshow(min_frame)
# plt.axis('off')
plt.imshow(min_frame[:,:,1])
plt.axis('off')
plt.tight_layout()
plt.savefig(save_fig_dir + "/" + env_name + "min_frame.png", bbox_inches='tight')