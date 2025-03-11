import cv2
import numpy as np
import gymnasium
from gymnasium import spaces
import math

class SoloPlayerEnv(gymnasium.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, server_url="localhost:23003", render_mode=""):
        super(SoloPlayerEnv, self).__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-math.pi,
            high=math.pi,
            shape=(2, 3, ), # position and rotation of the player
            dtype=np.float32
        )

        self.action_space = spaces.Tuple((
            spaces.Box(
                low=-math.pi,
                high=math.pi,
                shape=(3, ),
                dtype=np.float32
            ), # rotation of the player
            spaces.Discrete(1), # right click
            spaces.Discrete(1), # jump
            spaces.Discrete(1), # w
            spaces.Discrete(1), # a
            spaces.Discrete(1), # s
            spaces.Discrete(1), # d
            spaces.Discrete(1), # shift
            spaces.Discrete(1), # ctrl
        ))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)


        return obs, {}

    def step(self, action):
        return (
            obs,
            reward,
            terminated,
            truncated,
            {},
        )

    def close(self):
        pass


from gymnasium.envs.registration import register

register(
    id='solo-mc-speed-bridge-v0',
    entry_point='games.mc_speed_bridge.envs.solo_env:SoloPlayerEnv',
    max_episode_steps=300,
)