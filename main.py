import math
import sys
from enum import Enum
import copy

# Auth: Kieran Kim-Murphy

# direction enum
class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


# Return a potential position, given a direction of movement
def dir_to_pos(pos: tuple[int, int], dir: int) -> tuple[int, int]:
    dir = dir % 4
    if dir == 2:
        return pos[0] - 1, pos[1]
    elif dir == 0:
        return pos[0] + 1, pos[1]
    elif dir == 1:
        return pos[0], pos[1] - 1
    elif dir == 3:
        return pos[0], pos[1] + 1
    return pos[0], pos[1]


# Checks if a given square is safe to travel to
def is_valid_target(world: list[list[object]], target: tuple[int, int], height: int, width: int) -> bool:
    return (
            0 <= target[0] < width
            and 0 <= target[1] < height
            and world[target[0]][target[1]] is None
    )


# accept a suggested movement and correct it if a collision will occur
def safety_node(world: list[list[object]], pos: tuple[int, int], move: int) -> int:
    corrected_move = move
    target = dir_to_pos(pos, move)
    if not is_valid_target(world, target, WORLD_HEIGHT, WORLD_WIDTH):
        target = dir_to_pos(positions[p], move + 1)
        if is_valid_target(world, target, WORLD_HEIGHT, WORLD_WIDTH):
            corrected_move = move + 1
        else:
            corrected_move = move - 1
    return corrected_move


# calculate the manhattan distance between two points
def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


class Node:
    coordinates: tuple[int, int]
    distance: int
    parent: 'Node'

    def __init__(self, current_loc: tuple[int, int], dist: int, par: 'Node'):
        self.coordinates = current_loc
        self.distance = dist
        self.parent = par


def influence(world: list[list[object]], current_position: tuple[int, int]) -> list[list[object]]:
    q: list[Node] = []
    influence_map: list[list[object]] = [[None for _ in range(WORLD_HEIGHT)] for _ in range(WORLD_WIDTH)]
    current_node: Node = Node(current_position, 1, None)
    q.append(current_node)

    while q:
        current_node = q.pop(0)
        influence_map[current_node.coordinates[0]][current_node.coordinates[1]] = 1 / current_node.distance
        for i in range(4):
            p: tuple[int, int] = dir_to_pos(current_node.coordinates, i)
            if is_valid_target(world, p, WORLD_HEIGHT, WORLD_WIDTH) and influence_map[p[0]][p[1]] is None:
                influence_map[p[0]][p[1]] = 0
                q.append(Node(p, current_node.distance + 1, current_node))

    return influence_map


def subtract_influence(player: list[list[object]], enemy: list[list[object]], height, width) -> list[list[object], int]:
    influence_map: list[list[object]] = [[None for _ in range(WORLD_HEIGHT)] for _ in range(WORLD_WIDTH)]
    score: int = 0
    for i in range(width):
        for j in range(height):
            if player[i][j] is None:
                player[i][j] = 0
            if enemy[i][j] is None:
                enemy[i][j] = 0
            influence_map[i][j] = round(player[i][j] - enemy[i][j], 2)
            if influence_map[i][j] > 0:
                score += 1
            elif influence_map[i][j] < 0:
                score -= 1

    return influence_map, score


# calculate the value of a given board state using bfs floodfill as a heuristic
def find_value(world: list[list[object]], player: tuple[int, int], enemy: tuple[int, int]) -> int:
    player = influence(world, player)
    enemy = influence(world, enemy)
    _, score = subtract_influence(player, enemy, WORLD_HEIGHT, WORLD_WIDTH)
    return score


# prints the given world map to error
def print_map(world: list[list[object]]):
    temp: str = ""
    for row in world:
        temp = ""
        for val in row:
            val = str(val)
            temp += '{:6}'.format(val) + " "
        print(temp, file=sys.stderr, flush=True)


class MinimaxNode:
    state: list[list[object]]
    player_loc: tuple[int, int]
    player_num: int
    enemy_loc: tuple[int, int]
    enemy_num: int

    def __init__(self, world: list[list[object]], player: tuple[int, int], enemy: tuple[int, int], p: int, e: int):
        self.state = world
        self.player_loc = player
        self.player_num = p
        self.enemy_loc = enemy
        self.enemy_num = e


def simulate_move(world: list[list[object]], move: tuple[int, int], player_num: int) -> list[list[object]]:
    new_world: list[list[object]] = copy.deepcopy(world)
    new_world[move[0]][move[1]] = player_num
    return new_world


def minimax(node: MinimaxNode, depth: int, maximizing_player: bool) -> float:
    if depth <= 0:
        v = find_value(node.state, node.player_loc, node.enemy_loc)
        return v

    if maximizing_player:
        value = -math.inf
        for i in range(4):
            p: tuple[int, int] = dir_to_pos(node.player_loc, i)
            if is_valid_target(node.state, p, WORLD_HEIGHT, WORLD_WIDTH):
                child_state = simulate_move(node.state, p, node.player_num)
                child_node: MinimaxNode = MinimaxNode(child_state, p, node.enemy_loc, node.player_num, node.enemy_num)
                value = max(value, minimax(child_node, depth - 1, False))
        return value
    else:
        value = math.inf
        for i in range(4):
            e: tuple[int, int] = dir_to_pos(node.enemy_loc, i)
            if is_valid_target(node.state, e, WORLD_HEIGHT, WORLD_WIDTH):
                child_state = simulate_move(node.state, e, node.enemy_num)
                child_node: MinimaxNode = MinimaxNode(child_state, node.player_loc, e, node.player_num, node.enemy_num)
                value = min(value, minimax(child_node, depth - 1, True))
        return value

    return 0


def optimal_move(node: MinimaxNode, depth: int, maximizing_player: bool) -> int:
    optimal: int = 0
    value: float = -math.inf
    for i in range(4):
        p: tuple[int, int] = dir_to_pos(node.player_loc, i)
        if is_valid_target(node.state, p, WORLD_HEIGHT, WORLD_WIDTH):
            child_state = simulate_move(node.state, p, node.player_num)
            child_node: MinimaxNode = MinimaxNode(child_state, p, node.enemy_loc, node.player_num, node.enemy_num)
            m = minimax(child_node, depth - 1, False)
            if m > value:
                value = m
                optimal = i
            print(f"Optimal move in direction {i} at position ({p[0]}, {p[1]}) is {value}", file=sys.stderr, flush=True)
    return optimal


# global variables
WORLD_HEIGHT = 20
WORLD_WIDTH = 30
game_world = [[None for _ in range(WORLD_HEIGHT)] for _ in range(WORLD_WIDTH)]
move: int = 0

# game loop
while True:
    n, p = [int(i) for i in input().split()]  # n is number of players (2-4) and p is my assigned player num
    positions: list[tuple[int, int]] = []

    # Load in data
    for i in range(n):
        x0, y0, x1, y1 = [int(j) for j in input().split()]
        positions.append((x1, y1))
        game_world[x0][y0] = i
        game_world[x1][y1] = i

    # Plan with minimax
    m = optimal_move(MinimaxNode(game_world, positions[p], positions[n - 1 - p], p, n - 1 - p), 2, True)
    print(f"optimal move for player {p} is {m}", file=sys.stderr, flush=True)

    move = m

    # Avoid collisions
    move = safety_node(game_world, positions[p], move)

    # Normalize movement to Z%4
    move = move % 4

    board_value = find_value(game_world, positions[p], positions[n - 1 - p])
    print(f"Value of board for player {p} is {board_value}", file=sys.stderr, flush=True)

    # Issue move order
    print(Direction(move).name)
