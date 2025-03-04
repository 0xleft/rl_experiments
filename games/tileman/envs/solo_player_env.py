import cv2
import numpy as np
import gymnasium
from gymnasium import spaces
from .objects import Direction, Grid, Player, Tile, Vector, Game, Directions
import pygame

class SoloPlayerEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, grid_size=10, vision_range=5, max_steps=300, render_mode="rgb_array"):
        super(SoloPlayerEnv, self).__init__()
        self.render_mode = render_mode

        self.max_steps = max_steps
        self.vision_range = vision_range
        self.grid_size = grid_size
        self.game = Game(self.grid_size, self.grid_size)
        self.player = self.game.spawn_random_player()

        self.width = 600
        self.height = 600
        self.grid_tile_size = min(self.width // len(self.game.grid.tiles[0]), self.height // len(self.game.grid.tiles))
        self.screen = None
        self.clock = None

        self.action_space = spaces.Discrete(4)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1, 3 * (self.vision_range*2 + 1)**2),
            dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game = Game(self.grid_size, self.grid_size)
        self.player = self.game.spawn_random_player(seed=seed)

        if self.render_mode == "human":
            self._render_frame()

        return np.array([self.player.get_vision(self.game.grid, self.vision_range)]).astype(np.int8), {}  # empty info dict

    def step(self, action):
        if action not in Directions:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        
        previous_score = self.player.claim_count
        previous_position = self.player.position        

        self.player.move_direction = Directions[action]
        self.game.update()

        terminated = not self.player.is_alive
        truncated = False

        reward = 0.0
        
        if self.player.claim_count > previous_score:
            reward += 1.0
        if self.player.position == previous_position:
            reward -= 0.1

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()

        return (
            np.array([self.player.get_vision(self.game.grid, self.vision_range)]).astype(np.int8),
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
        
        for y, row in enumerate(self.game.grid.tiles):
            for x, tile in enumerate(row):
                pygame.draw.rect(canvas, (0, 0, 0), (x * self.grid_tile_size, y * self.grid_tile_size, self.grid_tile_size, self.grid_tile_size))
                if tile.claimed:
                    pygame.draw.rect(canvas, tile.claimer.color, (x * self.grid_tile_size, y * self.grid_tile_size, self.grid_tile_size, self.grid_tile_size))
                else:
                    pygame.draw.rect(canvas, (20, 20, 20), (x * self.grid_tile_size, y * self.grid_tile_size, self.grid_tile_size, self.grid_tile_size), 1)
                
                if tile.ocupied:
                    margin = self.grid_tile_size // 5
                    pygame.draw.rect(canvas, (max(tile.ocupant.color.r - 40, 0), max(tile.ocupant.color.g - 40, 0), max(tile.ocupant.color.b - 40, 0)), (x * self.grid_tile_size + margin, y * self.grid_tile_size + margin, self.grid_tile_size - 2 * margin, self.grid_tile_size - 2 * margin))
        
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
    id='tileman-solo-v0',
    entry_point='games.tileman.envs.solo_player_env:SoloPlayerEnv',
    max_episode_steps=300,
)