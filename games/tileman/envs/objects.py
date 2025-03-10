from typing import List
import pygame
import uuid
from copy import deepcopy
import time
import random
import numpy as np

class Vector:
    x: int
    y: int
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class Direction:
    UP = Vector(0, -1)
    DOWN = Vector(0, 1)
    LEFT = Vector(-1, 0)
    RIGHT = Vector(1, 0)

Directions = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT 
}

class Player:
    position: Vector
    color: pygame.Color
    move_direction: Direction = Direction.DOWN
    is_alive: bool = True
    id: uuid.UUID
    
    # specific neural network stuff
    kills: int = 0
    claim_count: int = 0
    max_claim_count: int = 0
    steps_survived: int = 0
    moves_since_capture: int = 0
    
    def __init__(self, x: int, y: int):
        self.position = Vector(x, y)
        self.color = pygame.Color(255, 255, 255)

        self.id = uuid.uuid4()

    def kill(self, grid: "Grid"):
        for row in grid.tiles:
            for tile in row:
                if tile.ocupant == self:
                    tile.unoccupy()
                if tile.claimer == self:
                    tile.unclaim()
                    
        self.is_alive = False

    def get_vision(self, grid: "Grid", vision_range: int = 20):
        left = self.position.x - vision_range
        right = self.position.x + vision_range
        top = self.position.y - vision_range
        bottom = self.position.y + vision_range
        
        vision = []
        
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                if (i < 0 or i >= grid.height) or (j < 0 or j >= grid.width):
                    vision.append((Tile(0, 0), True))
                    continue
                vision.append((deepcopy(grid.tiles[i][j]), False))

        ocupations = []
        claims = []
        borders = []
        
        for i in range((vision_range*2 + 1)**2):
            tile = vision[i][0]
            if tile.claimed:
                if tile.claimer.id == self.id:
                    claims.append(-1)
                else:
                    claims.append(1)
            else:
                claims.append(0)
            
            if tile.ocupied:
                if tile.ocupant.id == self.id:
                    ocupations.append(-1)
                else:
                    ocupations.append(1)
            else:
                ocupations.append(0)

            borders.append(1 if vision[i][1] else 0)
        
        result = np.array([claims, ocupations, borders]).astype(np.int8)
        result = result.reshape(3, 2 * vision_range + 1, 2 * vision_range + 1)
        return result


class Tile:
    position: Vector
    ocupied: bool
    ocupant: Player
    claimed: bool
    claimer: Player
    
    def __init__(self, x: int, y: int):
        self.position = Vector(x, y)
        self.ocupied = False
        self.claimed = False
        self.ocupant = None
        self.claimer = None

    def unclaim(self):
        self.claimed = False
        self.claimer = None
        
    def claim(self, player: Player):
        self.claimed = True
        self.claimer = player
        
    def unoccupy(self):
        self.ocupied = False
        self.ocupant = None
        
    def occupy(self, player: Player):
        self.ocupied = True
        self.ocupant = player


class Grid:
    tiles: List[List[Tile]]
    width: int
    height: int

    def __init__(self, width: int, height: int):
        self.tiles = [[Tile(x, y) for x in range(width)] for y in range(height)]
        self.width = width
        self.height = height
    
    def get_tile(self, x: int, y: int) -> Tile:
        return self.tiles[y][x]
    
    def get_tile_at(self, position: Vector) -> Tile:
        return self.tiles[position.y][position.x]
    

class Game:
    grid: Grid
    players: List[Player]
    width: int
    height: int
  
    def __init__(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.players = []
        self.width = width
        self.height = height

    def get_vision(self, player: Player, vision_range: int = 20):
        player_vision = player.get_vision(self.grid, vision_range)
        player_locations = np.zeros((1, 2 * vision_range + 1, 2 * vision_range + 1), dtype=np.int8)

        for other_player in self.players:
            if other_player == player:
                continue
            # check if player is outside of the vision range
            left = player.position.x - vision_range
            right = player.position.x + vision_range
            top = player.position.y - vision_range
            bottom = player.position.y + vision_range
            
            if other_player.position.x < left or other_player.position.x > right or other_player.position.y < top or other_player.position.y > bottom:
                continue
            
            player_locations[0][other_player.position.y - top, other_player.position.x - left] = -1

        return np.concatenate((player_vision, player_locations))

    def add_player(self, player: Player):
        self.players.append(player)
        # if player is on the border we move him inside by a square
        if player.position.x == 0:
            player.position.x += 1
        if player.position.x == len(self.grid.tiles[0]) - 1:
            player.position.x -= 1
        if player.position.y == 0:
            player.position.y += 1
        if player.position.y == len(self.grid.tiles) - 1:
            player.position.y -= 1
      
        # mark the 8 tiles around the player as claimed
        for y in range(-1, 2):
            for x in range(-1, 2):
                tile = self.grid.get_tile_at(Vector(player.position.x + x, player.position.y + y))
                tile.claim(player)

    def get_max_score(self) -> int:
        return max([player.claim_count for player in self.players]) if len(self.players) > 0 else 0

    def spawn_random_player(self, seed=None) -> Player:
        random.seed(seed)
        player = Player(random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        for other_player in self.players:
            if other_player.position == player.position:
                player = self.spawn_random_player(seed=seed)
                self.add_player(player)
                return player
                
        self.add_player(player)
        return player

    def update_player_move(self, player: Player):
        new_position = Vector(player.position.x + player.move_direction.x, player.position.y + player.move_direction.y)
        player.position = new_position
        tile = self.grid.get_tile_at(player.position)
        if tile.claimer != player:
            tile.occupy(player)
    
    def update_player_collisions(self, player: Player):
        new_position = Vector(player.position.x + player.move_direction.x, player.position.y + player.move_direction.y)
        if new_position.x < 0 or new_position.x >= len(self.grid.tiles[0]) or new_position.y < 0 or new_position.y >= len(self.grid.tiles):
            player.kill(self.grid)
            return
        
        tile = self.grid.get_tile_at(new_position)
        if tile.ocupied and not tile.claimed:
            tile.ocupant.kill(self.grid)
        
        if tile.ocupied and tile.claimed and tile.claimer != tile.ocupant:
            tile.ocupant.kill(self.grid)
  
    def update_player_claims(self, player: Player):
        player.moves_since_capture += 1

        new_position = Vector(player.position.x + player.move_direction.x, player.position.y + player.move_direction.y)
    
        if self.grid.get_tile_at(new_position).claimer == player and self.grid.get_tile_at(player.position).claimer != player:
            for row in self.grid.tiles:
                for tile in row:
                    if tile.ocupant == player:
                        if tile.claimer != None and tile.claimer != player:
                            tile.claimer.claim_count -= 1
                            
                        # to fill in the spaces in between the claim tiles we need to check for tiles until we find a tile that is going to be claimed (for right and left) then we claim all the tiles in between
                        tile.claim(player)
                        tile.unoccupy()
                        player.claim_count += 1
                        player.moves_since_capture = 0
                        
                        if player.claim_count > player.max_claim_count:
                            player.max_claim_count = player.claim_count
            
    def update_player_same_location(self, player: Player):
        # if players are in the same location we check if one of them is on a claim and the one that has claim wins if both are not on a claim both die or both are on a claim that neither of them posses they also both die
        for other_player in self.players:
            if other_player == player:
                continue

            if other_player.position == player.position:
                if self.grid.get_tile_at(player.position).claimed and not self.grid.get_tile_at(player.position).claimer == player:
                    other_player.kill(self.grid)
                elif self.grid.get_tile_at(player.position).claimed and not self.grid.get_tile_at(player.position).claimer == other_player:
                    player.kill(self.grid)
                else:
                    player.kill(self.grid)
                    other_player.kill(self.grid)
  
    def update(self):
        for player in self.players:
            self.update_player_collisions(player)
        
            if player.is_alive:
                player.steps_survived += 1
                
                self.update_player_claims(player)
                self.update_player_move(player)
                self.update_player_same_location(player)
        
        self.players = [player for player in self.players if player.is_alive]
    
        if len(self.players) == 0:
            return

        max_score = 0
        player_max_score = self.players[0]
        for player in self.players:
            if player.max_claim_count > max_score:
                max_score = player.claim_count
                player_max_score = player
            player.color = pygame.Color(230, 230, 230)
    
        if max_score > 0:
            player_max_score.color = pygame.Color(230, 0, 0)