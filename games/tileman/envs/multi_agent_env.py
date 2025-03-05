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

    async def handler(self, websocket, path=""):
        player = self.game.spawn_random_player()
        print(f"New player connected")
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
            await websocket.send(pickle.dumps(np.array([self.clients[websocket].get_vision(self.game.grid, self.vision_range)]).astype(np.int8)))
            return

        print(action) # todo remove debug
        player = self.clients[websocket]
        player.move_direction = Directions[action]

        self.players_moved[websocket] = True
        if all(self.players_moved.values()):
            self.players_moved = {ws: False for ws in self.players_moved}
            await self.send_observations()

    async def send_observations(self):
        before_update = {ws: {deepcopy(self.clients[ws])} for ws in self.clients.keys()}

        self.game.update()

        def calculate_reward(before_update_player: Player, player: Player):
            return player.claim_count - before_update_player.claim_count

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


    def __init__(self, host='localhost', port=9909):
        super(ClientPlayerEnv, self).__init__()
        
        self.host = host
        self.port = port
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.connect_to_server())

    async def connect_to_server(self):
        self.client = await websockets.connect(f"ws://{self.host}:{self.port}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.loop.run_until_complete(self.client.send(pickle.dumps("reset")))
        return self.loop.run_until_complete(self.client.recv())