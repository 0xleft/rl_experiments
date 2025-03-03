import numpy as np
import gymnasium
from gymnasium import spaces
from .objects import Direction, Grid, Player, Tile, Vector, Game, Directions
import pygame

class SoloPlayerEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}


    def __init__(self, grid_size=10, vision_range=5, max_steps=200, render_mode="human"):
        super(SoloPlayerEnv, self).__init__()
        self.render_mode = render_mode

        self.max_steps = max_steps
        self.vision_range = vision_range
        self.grid_size = grid_size
        self.game = Game(self.grid_size, self.grid_size)
        self.player = self.game.spawn_random_player()

        pygame.init()
        self.width = 500
        self.height = 500
        self.grid_tile_size = min(self.width // len(self.game.grid.tiles[0]), self.height // len(self.game.grid.tiles))
        self.screen = pygame.display.set_mode((self.width, self.height))


        self.action_space = spaces.Discrete(4)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1, 3, (self.vision_range*2 + 1)**2),
            dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game = Game(self.grid_size, self.grid_size)
        self.player = self.game.spawn_random_player(seed=seed)

        return np.array([self.player.get_vision(self.game.grid, self.vision_range)]).astype(np.int8), {}  # empty info dict

    def step(self, action):
        if action not in Directions:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        
        self.player.move_direction = Directions[action]
        self.game.update()

        terminated = not self.player.is_alive
        truncated = self.player.steps_survived >= self.max_steps

        reward = 0 # todo

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array([self.player.get_vision(self.game.grid, self.vision_range)]).astype(np.int8),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))
            for y, row in enumerate(self.game.grid.tiles):
                for x, tile in enumerate(row):
                    pygame.draw.rect(self.screen, (0, 0, 0), (x * self.grid_tile_size, y * self.grid_tile_size, self.grid_tile_size, self.grid_tile_size))
                    
                    if tile.claimed:
                        pygame.draw.rect(self.screen, tile.claimer.color, (x * self.grid_tile_size, y * self.grid_tile_size, self.grid_tile_size, self.grid_tile_size))
                    else:
                        pygame.draw.rect(self.screen, (20, 20, 20), (x * self.grid_tile_size, y * self.grid_tile_size, self.grid_tile_size, self.grid_tile_size), 1)
                    
                    if tile.ocupied:
                        margin = self.grid_tile_size // 5
                        pygame.draw.rect(self.screen, (max(tile.ocupant.color.r - 40, 0), max(tile.ocupant.color.g - 40, 0), max(tile.ocupant.color.b - 40, 0)), (x * self.grid_tile_size + margin, y * self.grid_tile_size + margin, self.grid_tile_size - 2 * margin, self.grid_tile_size - 2 * margin))
            

            pygame.display.flip()

    def close(self):
        pass