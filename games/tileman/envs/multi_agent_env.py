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
    def __init__(self, host='0.0.0.0', port=9909):
        self.host = host
        self.port = port
        self.clients = {}
        self.game = Game(10, 10)
        self.players_moved = {}

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

        

    def start(self):
        self.loop = asyncio.get_event_loop()
        server = websockets.serve(self.handler, self.host, self.port)
        self.loop.run_until_complete(server)
        self.loop.run_forever()

    def close(self):
        if self.loop is not None:
            self.loop.stop()

    @staticmethod
    def create_server():
        pass

class SoloPlayerEnv(gymnasium.Env):
    pass