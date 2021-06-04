import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
import matplotlib.pylab as plt
import argparse

from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
import pygame
import sys
import matplotlib
sys.path[0] += '/baselines'
from baselines.common.trex_utils import preprocess
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except Exception:
    pass

# parser = argparse.ArgumentParser(description=None)
# parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
# parser.add_argument('--reward_net_path', default='', help="name and location for learned model params")
# parser.add_argument('--seed', default=0, help="random seed for experiments")
# parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")
# parser.add_argument('--save_fig_dir', help ="where to save visualizations")


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Assume that we are on a CUDA machine, then this should print a CUDA device:
# print(device)


# args = parser.parse_args()
# env_name = args.env_name
# save_fig_dir = args.save_fig_dir

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

# seed = args.seed
# torch.manual_seed(seed)
# np.random.seed(seed)
# tf.set_random_seed(seed)

# print(env_id)

stochastic = True

# reward_net_path = args.reward_net_path

def make_env():
    def _thunk():
        # if repeat_action:
        #     env_name = "SpaceInvadersNoFrameskip-v0"
        # else:
        #     env_name = "SpaceInvadersNoFrameskip-v4"
        # parser = argparse.ArgumentParser(description=None)
        # parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
        # args = parser.parse_args()
        # env_name = args.env_name
        # if env_name == "spaceinvaders":
        #     env_id = "SpaceInvadersNoFrameskip-v4"
        # else:
        #     env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"
        env = gym.make("SpaceInvadersNoFrameskip-v4")
        # if noop_env:
        #     env = NoopResetEnv(env, noop_max=30)
        return env
    return _thunk


# env = make_env()
# env = DummyVecEnv([env])

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')

args = parser.parse_args()
env_name = args.env_name

if env_name == "spaceinvaders":
    env_id = "SpaceInvadersNoFrameskip-v4"

env = make_vec_env("SpaceInvadersNoFrameskip-v4", 'atari', 1, 1,
                   wrapper_kwargs={
                       'clip_rewards':False,
                       'episode_life':False,
                   })
env = VecFrameStack(env, 4)

agent = PPO2Agent(env, env_type, stochastic)

model_path = './baselines/col3_logdir_seed1/checkpoints/04000'
# model_path = './bc_models/col1.model.pth'

agent.load(model_path)
episode_count = 1
for i in range(episode_count):
    done = False
    r = 0
    ob = env.reset()
    #traj.append(ob)
    #print(ob.shape)
    while True:
        env.render()
        action = agent.act(ob, r, done)
        ob, r, done, _ = env.step(action)
        if done:
            break