import cv2
import numpy as np
import gymnasium
from gymnasium import spaces
import pygame

class LiveDataEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, starting_capital=1000.0, render_mode="rgb_array"): # starting capital is in euros
        super(LiveDataEnv, self).__init__()
        self.render_mode = render_mode

        self.capital = starting_capital
        self.action_space = spaces.Discrete(3) # buy, sell, do nothing

        self.observation_space = spaces.Tuple((spaces.Box( # the first box will represent the past stock prices for 1000 datapoints spanning over some time therefore it could 
            low=0,
            high=2**63 - 2, # max (stocks will prob never reach this price)
            shape=(1000, ), #
            dtype=np.float64
        ), spaces.Box(
           low=0,
           high=2**63 - 2,
           shape=(2, ), # represents how much of that stock we currently own[0] and how much capital we have[1]
           dtype=np.float64 
        )))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        

        return obs, {}  # empty info dict

    def step(self, action):
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.render_mode == "rgb_array":
            cv2.imshow('Window Name', self._render_frame())
            cv2.waitKey(1)
            return self._render_frame()
    
    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.width, self.height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.width, self.height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((0, 0, 0))
        
        # render logic in here todo

        self.surf = pygame.transform.flip(canvas, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


from gymnasium.envs.registration import register

register(
    id='stocks-live-data-v0',
    entry_point='games.stocks.envs.live_data:LiveDataEnv',
    max_episode_steps=None,
)