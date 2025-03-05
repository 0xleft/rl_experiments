import cv2
import numpy as np
import gymnasium
from gymnasium import spaces
from .objects import Direction, Grid, Player, Tile, Vector, Game, Directions
import pygame
import asyncio
import websockets
import pickle
import multiprocessing

class TileServer:
    def __init__(self, grid_size=40, vision_range=5, host='0.0.0.0', port=9909):
        self.host = host
        self.port = port
        self.grid_size = grid_size
        self.vision_range = vision_range
        self.clients = {}
        self.players_moved = {}
        self.game = Game(grid_size, grid_size)

    async def handler(self, websocket, path):
        player = self.game.spawn_random_player()
        self.clients[websocket] = player
        self.players_moved[websocket] = False
        try:
            async for message in websocket:
                # todo handle closing
                action = pickle.loads(message)
                await self.process_action(websocket, action)
        finally:
            del self.players_moved[websocket]
            del self.clients[websocket]
            player.kill(self.game.grid)

    async def process_action(self, websocket, action):
        if isinstance(action, str) and action == "close":
            self.close()
            return

        print(action) # todo remove debug
        player = self.clients[websocket]
        player.move_direction = Directions[action]

        self.players_moved[websocket] = True
        if all(self.players_moved.values()):
            self.game.update()
            self.players_moved = {ws: False for ws in self.players_moved}

            # send the new obs back
            # todo
            await asyncio.wait(websocket.send())
            

    async def send_observations(self):
        data = {ws: (
            np.array([self.clients[ws].get_vision(self.game.grid, self.vision_range)]).astype(np.int8),
            reward,
            terminated,
            truncated,
            info,
        ) for ws in self.clients}
        pickled_data = {
            ws: pickle.dumps(data[ws])
        for ws in self.clients}
        await asyncio.wait([ws.send(pickled_data[ws]) for ws in self.clients.keys()])
        self.moves.clear()

    def start(self):
        self.loop = asyncio.get_event_loop()
        server = websockets.serve(self.handler, self.host, self.port)
        self.loop.run_until_complete(server)
        self.loop.run_forever()

    def close(self):
        if self.loop is not None:
            self.loop.stop()

    @staticmethod
    def create_server(grid_size=40, vision_range=5, host='0.0.0.0', port=9909):
        # create new process
        server = TileServer(grid_size, vision_range, host, port)
        process = multiprocessing.Process()

class SoloPlayerEnv(gymnasium.Env):
    pass