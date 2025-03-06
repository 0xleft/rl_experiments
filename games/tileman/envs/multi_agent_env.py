from copy import deepcopy
import cv2
import numpy as np
import gymnasium
from gymnasium import spaces
from .objects import Direction, Grid, Player, Tile, Vector, Game, Directions
import pygame
import asyncio
import websockets
import pickle

class TileServer:
    def __init__(self, grid_size=40, vision_range=5, host='0.0.0.0', port=9909):
        self.host = host
        self.port = port
        self.grid_size = grid_size
        self.vision_range = vision_range
        self.clients: dict[websockets.ClientConnection, Player] = {}
        self.players_moved = {}
        self.game = Game(grid_size, grid_size)
        
        self.width = 600
        self.height = 600
        self.grid_tile_size = min(self.width // len(self.game.grid.tiles[0]), self.height // len(self.game.grid.tiles))

    async def handler(self, websocket, path=""):
        print("new client connected")
        player = self.game.spawn_random_player()
        self.clients[websocket] = player
        self.players_moved[websocket] = False
        try:
            async for message in websocket:
                action = pickle.loads(message)
                await self.process_action(websocket, action)
        finally:
            del self.players_moved[websocket]
            del self.clients[websocket]
            player.kill(self.game.grid)

    async def process_action(self, websocket: websockets.ClientConnection, action):
        if isinstance(action, str) and action == "close":
            self.close()
            return
        
        if isinstance(action, str) and action == "reset":
            self.clients[websocket].kill(self.game.grid)
            self.clients[websocket] = self.game.spawn_random_player()
            self.players_moved[websocket] = False
            await websocket.send(pickle.dumps(np.array([self.clients[websocket].get_vision(self.game.grid, self.vision_range)]).astype(np.int8)))
            return


        player = self.clients[websocket]
        player.move_direction = Directions[action]

        self.players_moved[websocket] = True

        print(self.players_moved.values())

        if all(self.players_moved.values()):
            self.players_moved = {ws: False for ws in self.players_moved}
            await self.send_observations()

    async def send_observations(self):
        before_update = {ws: deepcopy(self.clients[ws]) for ws in self.clients.keys()}

        self.game.update()
        # self.render()
        print("updating")

        def calculate_reward(before_update_player: Player, player: Player):
            return player.claim_count - before_update_player.claim_count + player.moves_since_capture * -0.01

        data = {ws: (
            np.array([self.clients[ws].get_vision(self.game.grid, self.vision_range)]).astype(np.int8),
            calculate_reward(before_update[ws], self.clients[ws]),
            not self.clients[ws].is_alive,
            False, # truncated
            {},
        ) for ws in self.clients.keys()}
        pickled_data = {
            ws: pickle.dumps(data[ws])
        for ws in self.clients.keys()}
        await asyncio.wait([ws.send(pickled_data[ws]) for ws in self.clients.keys()])

    async def start_server(self):
        print(f"Starting server at {self.host}:{self.port}")
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # run forever

    def start(self):
        asyncio.run(self.start_server())

    def close(self):
        if self.loop is not None:
            self.loop.stop()
            
    # def render(self):
    #     cv2.imshow('Window Name', self._render_frame())
    #     cv2.waitKey(1)
    # 
    # def _render_frame(self):
    #     canvas = pygame.Surface((self.width, self.height))
    #     canvas.fill((0, 0, 0))
    #     
    #     for y, row in enumerate(self.game.grid.tiles):
    #         for x, tile in enumerate(row):
    #             pygame.draw.rect(canvas, (0, 0, 0), (x * self.grid_tile_size, y * self.grid_tile_size, self.grid_tile_size, self.grid_tile_size))
    #             if tile.claimed:
    #                 pygame.draw.rect(canvas, tile.claimer.color, (x * self.grid_tile_size, y * self.grid_tile_size, self.grid_tile_size, self.grid_tile_size))
    #             else:
    #                 pygame.draw.rect(canvas, (20, 20, 20), (x * self.grid_tile_size, y * self.grid_tile_size, self.grid_tile_size, self.grid_tile_size), 1)
    #             if tile.ocupied:
    #                 margin = self.grid_tile_size // 5
    #                 pygame.draw.rect(canvas, (max(tile.ocupant.color.r - 40, 0), max(tile.ocupant.color.g - 40, 0), max(tile.ocupant.color.b - 40, 0)), (x * self.grid_tile_size + margin, y * self.grid_tile_size + margin, self.grid_tile_size - 2 * margin, self.grid_tile_size - 2 * margin))
    #     
    #     for player in self.game.players:
    #         margin = self.grid_tile_size // 5
    #         pygame.draw.rect(canvas, (255, 0, 0), (player.position.x * self.grid_tile_size + margin, player.position.y * self.grid_tile_size + margin, self.grid_tile_size - 2 * margin, self.grid_tile_size - 2 * margin))
# 
    #     self.surf = pygame.transform.flip(canvas, False, True)
    #     return np.transpose(
    #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #     )

    @staticmethod
    def create_server(grid_size=40, vision_range=5, host='0.0.0.0', port=9909):
        server = TileServer(grid_size, vision_range, host, port)
        server.start()

    @staticmethod
    def start_popen_process():
        import subprocess
        import sys
        import os
        import time

        process = subprocess.Popen([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'tile_server.py'))], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'tile_server.py')))
        time.sleep(1)
        return process


class ClientPlayerEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, vision_range=5, host='localhost', port=9909, render_mode="rgb_array"):
        super(ClientPlayerEnv, self).__init__()
        
        self.vision_range = vision_range
        self.host = host
        self.port = port
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1, 3 * (self.vision_range*2 + 1)**2),
            dtype=np.int8
        )
        
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.connect_to_server())

    async def connect_to_server(self):
        self.client = await websockets.connect(f"ws://{self.host}:{self.port}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.loop.run_until_complete(self.client.send(pickle.dumps("reset")))
        return pickle.loads(self.loop.run_until_complete(self.client.recv())), {}
    
    def step(self, action):
        self.loop.run_until_complete(self.client.send(pickle.dumps(action)))
        return pickle.loads(self.loop.run_until_complete(self.client.recv()))
    
from gymnasium.envs.registration import register

register(
    id='tileman-multi-v0',
    entry_point='games.tileman.envs.multi_agent_env:ClientPlayerEnv',
    max_episode_steps=300,
)