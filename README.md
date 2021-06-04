# Trajectory Ranked Imitation Learning

In my senior thesis, I use an inverse reinforcement learning method based on ranked demonstrations for imitation learning, which I call TRIL. This code base is adapted from the original code base for the inverse RL method called T-REX, located [here](https://github.com/hiwonjoon/ICML2019-TREX).

Most of my contributions are in the ``atari`` folder. The file ``play_traj.py`` is used to produce demonstration data, and may be easily modified to produce the desired file names of individual demonstrations. Then, ``LearnColRewards.py`` can be used to learn a reward function on the demonstration data, assuming that the demonstrations are ordered in increasing quality, and are generally stored in the ``learned_models`` folder. 

Use the instructions in the ``atari`` folder to run baselines to use regular RL on the learned reward functions. Visualizations of reward functions are in the ``atari/Visualizations`` folder, and were produced by ``visualize_reward.py``.
