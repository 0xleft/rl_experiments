import time
import nest_asyncio
nest_asyncio.apply()
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
import threading
import subprocess

class TileServerLoadBalancer:
    def __init__(self, max_players_per_server=4, grid_size=40, vision_range=5, host='localhost', port=32544):
        self.host = host
        self.port = port
        self.grid_size = grid_size
        self.vision_range = vision_range
        self.max_players_per_server = max_players_per_server
        self.next_port = port + 1

        self.servers: dict[subprocess.Popen, tuple[list[websockets.ClientConnection], int]] = {}
        self.create_new_server() # the default one server

    async def start_server(self):
        print(f"Starting load balancer on {self.host}:{self.port}")
        async with websockets.serve(self.new_client, "0.0.0.0", self.port):
            await asyncio.Future()

    def start(self):
        asyncio.run(self.start_server())

    def get_good_server(self) -> int:
        for server in self.servers.keys():
            if len(self.servers[server][0]) < self.max_players_per_server:
                return self.servers[server][1]
            
        # create a new server
        self.create_new_server()
        return self.next_port - 1

    async def new_client(self, websocket, path=""):
        good_server_port = self.get_good_server()
        
        print(f"Found good server on ws://{self.host}:{good_server_port}")
        
        for _, (clients, port) in self.servers.items():
            if port == good_server_port:
                clients.append(websocket)
                break
            
        async with websockets.connect(f"ws://localhost:{good_server_port}") as ws:
            async def proxy_forward():
                async for message in websocket:
                    await ws.send(message)
            async def proxy_backward():
                async for message in ws:
                    await websocket.send(message)
                    
            await asyncio.gather(proxy_forward(), proxy_backward())

    def create_new_server(self):
        server = TileServer.start_popen_process(port=self.next_port)
        self.servers[server] = ([], self.next_port)
        self.next_port += 1
        print(f"Created new child server on port {self.next_port - 1}")
        return server

    def clean_up_servers(self):
        # if there are any empty servers close them
        pass

    def close(self):
        for server in self.servers.keys():
            server.kill()

class TileServer:
    def __init__(self, grid_size=20, vision_range=5, host='0.0.0.0', port=9909):
        self.host = host
        self.port = port
        self.grid_size = grid_size
        self.ignore_task = None
        self.vision_range = vision_range
        self.clients: dict[websockets.ClientConnection, dict] = {
            # websocket: {
            #     "player": Player,
            #     "moved": bool,
            #     "should_ignore": bool,
            #     "time_since_last_move": int,
            #     "is_resetting": bool,
            # }
        }
        self.game = Game(grid_size, grid_size)
        self.checking_for_ignore = False
        
        self.width = 600
        self.height = 600
        self.grid_tile_size = min(self.width // len(self.game.grid.tiles[0]), self.height // len(self.game.grid.tiles))

        self.running = True
        self.render_thread = threading.Thread(target=self.render_loop)
        self.render_thread.start()

    def render_loop(self):
        while self.running:
            self.render()
            time.sleep(1/60)

    async def handler(self, websocket, path=""):
        print("new client connected")
        player = self.game.spawn_random_player()
        self.clients[websocket] = {}
        self.clients[websocket]["player"] = player
        self.clients[websocket]["moved"] = False
        self.clients[websocket]["is_resetting"] = False
        self.clients[websocket]["should_ignore"] = False
        try:
            async for message in websocket:
                try:
                    action = pickle.loads(message)
                except pickle.UnpicklingError:
                    print(message)
                    continue
                await self.process_action(websocket, action)
        except websockets.ConnectionClosedError:
            pass
        finally:
            del self.clients[websocket]
            player.kill(self.game.grid)

    async def process_action(self, websocket: websockets.ClientConnection, action):
        if isinstance(action, str) and action == "close":
            self.close()
            return
        
        if isinstance(action, str) and action == "keepalive":
            return
        
        if isinstance(action, str) and action == "reset":
            self.clients[websocket]["player"].kill(self.game.grid)
            self.clients[websocket]["player"] = self.game.spawn_random_player()
            self.clients[websocket]["is_resetting"] = True
            self.clients[websocket]["moved"] = False
            self.clients[websocket]["should_ignore"] = False
            await websocket.send(pickle.dumps(self.game.get_vision(self.clients[websocket]["player"], self.vision_range)))
            # self.render()
            return
        
        self.clients[websocket]["is_resetting"] = False
        self.clients[websocket]["should_ignore"] = False
        self.clients[websocket]["player"].move_direction = Directions[action]
        self.clients[websocket]["moved"] = True

        # print(list(self.clients[ws]["should_ignore"] for ws in self.clients))
        # if most clients have moved place a timer that after some time if no move is done sets ignore to false to the clients that have not moved
        
        async def set_should_ignore():
            await asyncio.sleep(1)

            for ws in self.clients:
                if not self.clients[ws]["moved"]:
                    self.clients[ws]["should_ignore"] = True
            
            await self.check_should_update()
            self.checking_for_ignore = False

        if not self.checking_for_ignore:
            self.checking_for_ignore = True
            self.ignore_task = asyncio.create_task(set_should_ignore())
        else:
            await self.check_should_update()

    async def check_should_update(self):
        if all(self.clients[ws]["moved"] or self.clients[ws]["should_ignore"] for ws in self.clients):
            if self.ignore_task:
                self.ignore_task.cancel()
                self.ignore_task = None
            for ws in self.clients:
                self.clients[ws]["moved"] = False
            await self.send_observations()

    async def send_observations(self):
        before_update = {ws: deepcopy(self.clients[ws]["player"]) for ws in self.clients.keys()}

        self.game.update()
        # self.render()

        def calculate_reward(before_update_player: Player, player: Player):
            if not player.is_alive:
                return -1
            reward = (player.claim_count - before_update_player.claim_count) * 0.9 + (player.kills - before_update_player.kills) * 5
            return min(5, max(-5, reward)) # clip between 5 and -5

        data = {ws: (
            self.game.get_vision(self.clients[ws]["player"], self.vision_range),
            calculate_reward(before_update[ws], self.clients[ws]["player"]),
            not self.clients[ws]["player"].is_alive,
            False, # truncated
            {},
        ) for ws in self.clients.keys()}
        pickled_data = {
            ws: pickle.dumps(data[ws])
        for ws in self.clients.keys()}
        await asyncio.wait([ws.send(pickled_data[ws]) for ws in self.clients.keys() if not self.clients[ws]["is_resetting"] and not self.clients[ws]["should_ignore"]])

    async def start_server(self):
        print(f"Starting server at {self.host}:{self.port}")
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # run forever

    def start(self):
        asyncio.run(self.start_server())

    def close(self):
        if self.loop is not None:
            self.loop.stop()
            self.running = False
            self.render_thread.join()
            self.ignore_task.cancel()
            
    def render(self):
        cv2.imshow('Window Name', self._render_frame())
        cv2.waitKey(1)

    def _render_frame(self):
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
        for player in self.game.players:
            margin = self.grid_tile_size // 5
            pygame.draw.rect(canvas, (255, 0, 0), (player.position.x * self.grid_tile_size + margin, player.position.y * self.grid_tile_size + margin, self.grid_tile_size - 2 * margin, self.grid_tile_size - 2 * margin))
        self.surf = pygame.transform.flip(canvas, False, True)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    @staticmethod
    def create_server(grid_size=40, vision_range=5, host='0.0.0.0', port=9909):
        server = TileServer(grid_size, vision_range, host, port)
        server.start()
        return server

    @staticmethod
    def start_popen_process(port=9909):
        import subprocess
        import sys
        import os
        import time

        def print_output(process):
            def print_pipe(pipe):
                for line in iter(pipe.readline, b''):
                    print(line.decode(), end='')
            threading.Thread(target=print_pipe, args=(process.stdout,), daemon=True).start()
            threading.Thread(target=print_pipe, args=(process.stderr,), daemon=True).start()

        process = subprocess.Popen([sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'tile_server.py')), f"{port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'tile_server.py')), port)
        print_output(process)
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
            shape=(4, (self.vision_range*2 + 1), (self.vision_range*2 + 1)),
            dtype=np.int8
        )
        
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.connect_to_server())

        # self.start_keepalive()
    
    def start_keepalive(self):
        self.running = True

        async def keepalive():
            while self.running:
                await asyncio.sleep(5)
                try:
                    await self.client.send(pickle.dumps("keepalive"))
                except:
                    return

        self.keep_alive_thread = threading.Thread(target=lambda: asyncio.run(keepalive()))
        self.keep_alive_thread.start()
    
    async def connect_to_server(self):
        self.client = await websockets.connect(f"ws://{self.host}:{self.port}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        # self.loop.run_until_complete(self.client.close())
        # self.loop.run_until_complete(self.connect_to_server())
        
        self.loop.run_until_complete(self.client.send(pickle.dumps("reset")))
        return pickle.loads(self.loop.run_until_complete(self.client.recv())), {}
        
    def step(self, action):
        self.loop.run_until_complete(self.client.send(pickle.dumps(action)))
        return pickle.loads(self.loop.run_until_complete(self.client.recv()))
    
    def close(self):
        self.loop.run_until_complete(self.client.close())
        self.running = False
        self.keep_alive_thread.join()
    
from gymnasium.envs.registration import register

register(
    id='tileman-multi-v0',
    entry_point='games.tileman.envs.multi_agent_env:ClientPlayerEnv',
    max_episode_steps=300,
)