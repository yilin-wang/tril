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


import pyglet.window as pw

from collections import deque
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from threading import Thread

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    pyg_img = pyg_img.copy()
    screen.blit(pyg_img, (0,0))

def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    """Allows one to play the game using keyboard.

    To simply play the game use:

        play(gym.make("Pong-v3"))

    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.

        def callback(obs_t, obs_tp1, rew, done, info):
            return [rew,]
        env_plotter = EnvPlotter(callback, 30 * 5, ["reward"])

        env = gym.make("Pong-v3")
        play(env, callback=env_plotter.callback)


    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environemnt is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """

    obs_s = env.observation_space
    assert type(obs_s) == gym.spaces.box.Box
    #assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1,3])


    # if keys_to_action is None:
    #     if hasattr(env, 'get_keys_to_action'):
    #         keys_to_action = env.get_keys_to_action()
    #     elif hasattr(env.unwrapped, 'get_keys_to_action'):
    #         keys_to_action = env.unwrapped.get_keys_to_action()
    #     else:
    #         assert False, env.spec.id + " does not have explicit key to action mapping, " + \
    #                       "please specify one manually"

    keys_to_action = {(): 0, (pygame.locals.K_SPACE,): 1, (pygame.locals.K_RIGHT,): 2, (pygame.locals.K_LEFT,): 3, (pygame.locals.K_SPACE, pygame.locals.K_RIGHT): 4, (pygame.locals.K_SPACE, pygame.locals.K_LEFT): 5}
    relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))

    if transpose:
        video_size = env.observation_space.shape[1], env.observation_space.shape[0]
    else:
        video_size = env.observation_space.shape[0], env.observation_space.shape[1]

    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    # Trajectory as list of observations
    ob_traj = []
    ac_traj = []
    steps = 0

    # Demonstrations as list of trajectories
    # demonstrations = []
    dones = 0

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            # Timer
            print(pygame.time.get_ticks())
            action = keys_to_action[tuple(sorted(pressed_keys))]
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)
            steps += 1
            if pygame.time.get_ticks() > 0:
                # Preprocess obs, append observations to trajectory
                # obs1 = cv2.cvtColor(obs[:,:,:3], cv2.COLOR_BGR2RGB)
                # obs1 = cv2.cvtColor(obs1, cv2.COLOR_RGB2GRAY)
                # obs1 = cv2.resize(obs1, (84, 84), interpolation=cv2.INTER_AREA)
                # obs2 = cv2.cvtColor(obs[:,:,3:6], cv2.COLOR_BGR2RGB)
                # obs2 = cv2.cvtColor(obs2, cv2.COLOR_RGB2GRAY)
                # obs2 = cv2.resize(obs2, (84, 84), interpolation=cv2.INTER_AREA)
                # obs3 = cv2.cvtColor(obs[:,:,6:9], cv2.COLOR_BGR2RGB)
                # obs3 = cv2.cvtColor(obs3, cv2.COLOR_RGB2GRAY)
                # obs3 = cv2.resize(obs3, (84, 84), interpolation=cv2.INTER_AREA)
                # obs4 = cv2.cvtColor(obs[:,:,-3:], cv2.COLOR_BGR2RGB)
                # obs4 = cv2.cvtColor(obs4, cv2.COLOR_RGB2GRAY)
                # obs4 = cv2.resize(obs4, (84, 84), interpolation=cv2.INTER_AREA)
                # obs = np.stack((obs1,obs2,obs3,obs4),axis=-1)
                ob_processed = preprocess(obs,"spaceinvaders")
                ob_processed = ob_processed[0] #get rid of first dimension ob.shape = (1,84,84,4)
                ob_traj.append(ob_processed)
                ac_traj.append(action)
            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info)
            if env_done == True:
                dones += 1
                # demonstrations.append(traj)
                # pygame.quit()
            if dones >= 12:
                pygame.quit()
                with open('bc_demos/obs_row1', 'wb') as fp:
                    pickle.dump(ob_traj, fp)
                with open('bc_demos/acs_row1', 'wb') as fp:
                    pickle.dump(ac_traj, fp)
                running = False
            #     pygame.quit()
            #     with open('col_demonstrations', 'wb') as fp:
            #         pickle.dump(demonstrations, fp)
        if running == False:
            break
        if obs is not None:
            # if len(obs.shape) == 2:
            #     obs = obs[:, :, None]
            # if obs.shape[2] == 1:
            #     obs = obs.repeat(3, axis=2)
            #rendered=env.envs[0].render(mode='rgb_array')
            rendered=env.render(mode='rgb_array')
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()

class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data     = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]))
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.000001)

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

if __name__ == '__main__':
    #env = gym.make("MontezumaRevengeNoFrameskip-v4")
    env = make_vec_env("SpaceInvadersNoFrameskip-v4", 'atari', 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })
    # env = gym.make("SpaceInvadersNoFrameskip-v4")
    # env = DummyVecEnv([env])

    # env = make_env()
    # env = DummyVecEnv([env])
    env = VecFrameStack(env, 4)

    # ob_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
    # shp = ob_space.shape
    # ob_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * 4), dtype=np.uint8)
    # ac_space = env.action_space
    play(env, zoom=4, fps=15)
    # with open('col_demonstrations', 'rb') as fp:
        # demonstrations = pickle.load(fp)