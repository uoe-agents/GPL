import copy
import random
import pickle as pkl
import numpy as np
import threading
import gym
from gym import spaces
import time
import sys, os

from Wolfpack_gym.envs.wolfpack_single_assets.Agents import *

"""
Class that generates random gridworld with obstacles for Wolfpack. 
Implementation based on the cellular automata algorithm. Fiddle with
the deathLimit and birthLimit parameters to change the density of the
resulting obstacles.
"""


class Generator(object):
    def __init__(self, size, deathLimit=4, birthLimit=3):
        self.x_size = size[0]
        self.y_size = size[1]
        self.booleanMap = [[False] * k for k in [self.x_size] * self.y_size]
        self.probStartAlive = 0.82;
        self.deathLimit = deathLimit
        self.birthLimit = birthLimit
        self.copy = None

    def initialiseMap(self):
        for x in range(self.x_size):
            for y in range(self.y_size):
                if random.random() < self.probStartAlive:
                    self.booleanMap[y][x] = True

    def doSimulationStep(self):

        newMap = [[False] * k for k in [self.x_size] * self.y_size]
        for x in range(self.x_size):
            for y in range(self.y_size):
                alive = self.countAliveNeighbours(x, y)
                if self.booleanMap[y][x]:
                    if alive < self.deathLimit:
                        newMap[y][x] = False
                    else:
                        newMap[y][x] = True
                else:
                    if alive > self.birthLimit:
                        newMap[y][x] = True
                    else:
                        newMap[y][x] = False
        self.booleanMap = newMap

    def countAliveNeighbours(self, x, y):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbour_x = x + i
                neighbour_y = y + j
                if not ((i == 0) and (j == 0)):
                    if neighbour_x < 0 or neighbour_y < 0 or neighbour_x >= self.x_size or neighbour_y >= self.y_size:
                        count = count + 1
                    elif self.booleanMap[neighbour_y][neighbour_x]:
                        count = count + 1
        return count

    def simulate(self, numSteps):
        done = False
        while not done:
            self.booleanMap = [[False] * k for k in [self.x_size] * self.y_size]
            self.initialiseMap()
            for kk in range(numSteps):
                self.doSimulationStep()

            if self.doFloodfill(self.booleanMap):
                done = True

    def doFloodfill(self, newMap):
        self.copy = copy.deepcopy(newMap)
        foundX, foundY = -1, -1
        for i in range(len(self.copy)):
            flag = False
            for j in range(len(self.copy[i])):
                if not self.copy[i][j]:
                    foundX = i
                    foundY = j
                    flag = True
                    break
            if flag:
                break
        self.floodfill(foundX, foundY)
        done = True
        for i in range(len(self.copy)):
            flag = False
            for j in range(len(self.copy[i])):
                # print(self.copy[i][j])
                if not self.copy[i][j]:
                    done = False
                    flag = True
                    break
            if flag:
                break
        return done

    def floodfill(self, x, y):
        queue = []
        queue.append((x, y))
        while len(queue) != 0:
            a = queue[0][0]
            b = queue[0][1]

            del queue[0]
            if not self.copy[a][b]:
                self.copy[a][b] = True

            if (not a + 1 >= len(self.copy)) and (not self.copy[a + 1][b]):
                queue.append((a + 1, b))
                self.copy[a + 1][b] = True
            if (not (a - 1 < 0)) and (not self.copy[a - 1][b]):
                queue.append((a - 1, b))
                self.copy[a - 1][b] = True
            if (not b + 1 >= len(self.copy[0])) and (not self.copy[a][b + 1]):
                queue.append((a, b + 1))
                self.copy[a][b + 1] = True
            if (not b - 1 < 0) and (not self.copy[a][b - 1]):
                queue.append((a, b - 1))
                self.copy[a][b - 1] = True


class WolfpackSingle(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_height=20, grid_width=20, num_players=5, max_food_num=2,
                 sight_sideways=8, sight_radius=8, max_time_steps=200,
                 coop_radius=1, groupMultiplier=2, food_freeze_rate=0, seed=None,
                 obs_type="main_player", with_random_grid=False, random_grid_dir=None,
                 prey_with_gpu=False, with_oppo_mod=True):

        # Define width and height of grid world
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.obs_type = obs_type
        self.with_oppo_mod = with_oppo_mod
        self.num_players = num_players

        N_DISCRETE_ACTIONS = 5
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        box_high = []
        for idx in range(num_players):
            box_high.append(self.grid_height - 1)
            box_high.append(self.grid_width - 1)

        for idx in range(6 * max_food_num):
            if idx % 6 == 0:
                box_high.append(self.grid_height - 1)
            elif idx % 6 == 1:
                box_high.append(self.grid_width-1)
            else:
                box_high.append(1)

        box_low = [0] * len(box_high)

        if self.with_oppo_mod:
            box_high.extend([N_DISCRETE_ACTIONS-1] * (num_players-1))
            box_low.extend([0] * (num_players-1))

        self.observation_space = spaces.Box(low=np.array(box_low), high=np.array(box_high),
                                                dtype=np.int32)

        # Define seeds for the agent initial position generator
        if seed is None:
            seed = int(time.time())
        self.seed = seed
        # self.randomizer = random
        np.random.seed(self.seed)

        # Define the observation space of the preys
        self.sight_sideways = sight_sideways
        self.sight_radius = sight_radius
        self.pads = max(self.sight_sideways, self.sight_radius)

        # Padded RGB matrix for preys that receive RGB inputs
        self.RGB_padded_grid = [[[0, 0, 255] for b in range(2 * self.pads + self.grid_width)]
                                for _ in range(2 * self.pads + self.grid_height)]

        # Grid to save the locations of agents, preys and obstacles
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]

        if "full_rgb" in self.obs_type:
            # RGB grid without padding in case agents use RGB grids as input
            self.RGB_grid = [[[0, 0, 255] for b in range(self.grid_width)]
                             for _ in range(self.grid_height)]

        # Initialize players and preys
        self.num_players = num_players
        self.max_food_num = max_food_num

        # Initialize maximum episode length and prey cooldown period
        self.max_time_steps = max_time_steps
        self.food_freeze_rate = food_freeze_rate

        self.other_player_acts = None
        self.other_player_obses = None

        # Use randomly generated grid or not
        self.with_random_grid = with_random_grid
        self.random_grid_dir = random_grid_dir

        if not self.random_grid_dir is None:
            self.levelMap = self.load_map(self.random_grid_dir)
        elif not self.with_random_grid:
            self.levelMap = [[False] * k for k in [self.grid_width] * self.grid_height]
        else:
            app = Generator((self.grid_width, self.grid_height), 7, 8)
            app.initialiseMap()
            app.simulate(2)
            self.levelMap = app.booleanMap

        self.visualizer = None

        self.obstacleCoord = [(iy, ix) for ix, row in enumerate(self.levelMap) for iy, i in enumerate(row) if i]
        self.possibleCoordinates = None
        self.player_positions = None
        self.player_orientation = None
        self.food_positions = None
        self.food_alive_statuses = None
        self.food_frozen_time = None
        self.food_orientation = None
        self.player_points = None
        self.food_points = None
        self.a_to_idx = None
        self.idx_to_a = None
        self.prev_dist_to_food = None

        self.food_obs_type = ["partial_obs" for _ in range(self.max_food_num)]
        self.food_obses = None
        self.prey_with_gpu = prey_with_gpu

        self.remaining_timesteps = max_time_steps
        self.food_list = [DQNAgent(agent_id=a, args={"with_gpu": self.prey_with_gpu, "max_seq_length": 5},
                                   obs_type="partial_obs")
                          for a in range(self.max_food_num)]

        dirname = os.path.dirname(__file__)
        for idx, agent in enumerate(self.food_list):
            filename = os.path.join(dirname,
                                    ("wolfpack_assets/dqn_prey_parameters/exp0.0001param_10_agent_" + str(idx)))
            agent.load_parameters(filename)

        self.other_agents_list = self.sample_teammates(self.num_players - 1)
        self.player_obs_type = [self.obs_type]
        self.player_obs_type.extend([agent.obs_type for agent in self.other_agents_list])

        self.sight_sideways = sight_sideways
        self.sight_radius = sight_radius
        self.coopRadius = coop_radius
        self.groupMultiplier = groupMultiplier

    def save_map(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.levelMap, f)

    def load_map(self, filename):
        with open(filename, 'rb') as f:
            self.levelMap = pkl.load(f)

    def reset(self):
        # Reset Visualizer
        self.seed = self.seed+1250
        np.random.seed(self.seed)
        self.visualizer = None

        # Reset padded grids
        self.RGB_padded_grid = [[[0, 0, 255] for _ in range(2 * self.pads + self.grid_width)]
                                for a in range(2 * self.pads + self.grid_height)]

        # Reset grid locations
        self.grid = [[0 for b in range(self.grid_width)] for a in range(self.grid_height)]

        if "full_rgb" in self.obs_type:
            # Reset RGB Grid
            self.RGB_grid = [[[0, 0, 255] for b in range(self.grid_width)] for a in range(self.grid_height)]

        self.possibleCoordinates = [(iy, ix) for ix, row in enumerate(self.levelMap) for iy, i in enumerate(row) if
                                    not i]

        # Sample initial location
        player_loc_idx = np.random.choice(range(len(self.possibleCoordinates)), self.num_players, replace=False)
        # player_loc_idx = random.sample(range(len(self.possibleCoordinates)), self.num_players)
        self.player_positions = [self.possibleCoordinates[a] for a in player_loc_idx]

        # Reset initial player orientation and points
        self.player_orientation = [0 for a in range(self.num_players)]
        self.player_points = [0 for a in range(self.num_players)]

        self.other_player_acts = None

        coordinates_no_player = [a for a in self.possibleCoordinates if a not in self.player_positions]
        food_loc_idx = np.random.choice(range(len(coordinates_no_player)), self.max_food_num, replace=False)
        # food_loc_idx = random.sample(range(len(coordinates_no_player)), self.max_food_num)

        # Reset all food attributes
        self.food_obses = None
        self.food_positions = [coordinates_no_player[a] for a in food_loc_idx]
        self.food_alive_statuses = [True for a in range(self.max_food_num)]
        self.food_frozen_time = [0 for a in range(self.max_food_num)]
        self.food_points = [0 for a in range(self.max_food_num)]

        self.food_orientation = [0 for a in range(self.max_food_num)]

        for coord in self.possibleCoordinates:
            self.grid[coord[0]][coord[1]] = 1
            if "full_rgb" in self.obs_type:
                self.RGB_grid[coord[0]][coord[1]] = [0, 0, 0]
            self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [0, 0, 0]
        for coord in self.player_positions:
            self.grid[coord[0]][coord[1]] = 2
            if "full_rgb" in self.obs_type:
                self.RGB_grid[coord[0]][coord[1]] = [255, 255, 255]
            self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [255, 255, 255]
        for coord in self.food_positions:
            self.grid[coord[0]][coord[1]] = 3
            if "full_rgb" in self.obs_type:
                self.RGB_grid[coord[0]][coord[1]] = [255, 0, 0]
            self.RGB_padded_grid[coord[0] + self.pads][coord[1] + self.pads] = [255, 0, 0]

        self.prev_dist_to_food = [min([abs(px - fx) + abs(py - fy) for (fx, fy) in self.food_positions])
                                  for (px, py) in self.player_positions]
        self.remaining_timesteps = self.max_time_steps

        self.other_agents_list = self.sample_teammates(self.num_players - 1)
        print(self.other_agents_list)
        self.player_obs_type = [self.obs_type]
        self.player_obs_type.extend([agent.obs_type for agent in self.other_agents_list])

        player_obses = [self.observation_computation(obs_type, agent_id=id) for id, obs_type in
                        enumerate(self.player_obs_type)]
        food_obses = [self.observation_computation(obs_type,
                                                   agent_type="food", agent_id=id) for id, obs_type in
                      enumerate(self.food_obs_type)]
        self.food_obses = food_obses
        self.other_player_obses = player_obses[1:]

        self.food_list = [DQNAgent(agent_id=a, args={"with_gpu": self.prey_with_gpu, "max_seq_length": 5},
                                   obs_type="partial_obs")
                          for a in range(self.max_food_num)]
        dirname = os.path.dirname(__file__)
        for idx, agent in enumerate(self.food_list):
            filename = os.path.join(dirname,
                                    ("wolfpack_assets/dqn_prey_parameters/exp0.0001param_10_agent_" + str(idx)))
            agent.load_parameters(filename)

        return player_obses[0]

    def revive(self):
        # find possible locations to revive dead prey
        coordinates_no_player = [a for a in self.possibleCoordinates if
                                 a not in self.player_positions and a not in self.food_positions]
        revived_idxes = []
        for idx, food in enumerate(self.food_positions):
            if self.food_frozen_time[idx] <= 0 and not self.food_alive_statuses[idx]:
                revived_idxes.append(idx)

        if len(revived_idxes) > 0:
            idxes = []
            for k in range(len(revived_idxes)):
                idx = np.random.choice(range(len(coordinates_no_player)), 1)[0]
                # idx = random.sample(range(len(coordinates_no_player)), 1)[0]
                while idx in idxes:
                    idx = np.random.choice(range(len(coordinates_no_player)), 1)[0]
                    # idx = random.sample(range(len(coordinates_no_player)), 1)[0]
                idxes.append(idx)
            coords = [coordinates_no_player[idx] for idx in idxes]

            coord_idx = 0
            for idx in revived_idxes:
                self.food_alive_statuses[idx] = True
                self.food_positions[idx] = coords[coord_idx]
                coord_idx += 1

        self.prev_dist_to_food = [min([abs(px - fx) + abs(py - fy) for (fx, fy) in self.food_positions])
                                  for (px, py) in self.player_positions]

    def update_status(self):
        for idx in range(len(self.food_alive_statuses)):
            if not self.food_alive_statuses[idx]:
                self.food_frozen_time[idx] -= 1

    def calculate_new_position(self, collectiveAct, prev_player_position, prev_player_orientation):
        zipped_data = list(zip(collectiveAct, prev_player_position, prev_player_orientation))
        result = [self.calculate_indiv_position(a, (b, c), d) for (a, (b, c), d) in zipped_data]
        return result

    def calculate_indiv_position(self, action, pair, orientation):
        x = pair[0]
        y = pair[1]
        next_x = x
        next_y = y

        # go forward
        if action == 0:
            # Facing upwards
            if orientation == 0:
                next_x -= 1
            # Facing right
            elif orientation == 1:
                next_y += 1
            # Facing downwards
            elif orientation == 2:
                next_x += 1
            else:
                next_y -= 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)
        # Step right
        elif action == 1:
            # Facing upwards
            if orientation == 0:
                next_y += 1
            # Facing right
            elif orientation == 1:
                next_x += 1
            # Facing downwards
            elif orientation == 2:
                next_y -= 1
            else:
                next_x -= 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)
        # Step back
        elif action == 2:
            # Facing upwards
            if orientation == 0:
                next_x += 1
            # Facing right
            elif orientation == 1:
                next_y -= 1
            # Facing downwards
            elif orientation == 2:
                next_x -= 1
            else:
                next_y += 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)
        # Step left
        elif action == 3:
            # Facing upwards
            if orientation == 0:
                next_y -= 1
            # Facing right
            elif orientation == 1:
                next_x -= 1
            # Facing downwards
            elif orientation == 2:
                next_y += 1
            else:
                next_x += 1

            if (next_x, next_y) in set(self.possibleCoordinates):
                return (next_x, next_y, orientation)
            else:
                return (x, y, orientation)
        # stay still
        elif action == 4:
            return (x, y, orientation)
        # rotate left
        elif action == 5:
            new_orientation = 0
            if orientation == 0:
                new_orientation = 3
            elif orientation == 1:
                new_orientation = 0
            elif orientation == 2:
                new_orientation = 1
            else:
                new_orientation = 2

            return (x, y, new_orientation)
        # rotate right
        else:
            new_orientation = 0
            if orientation == 0:
                new_orientation = 1
            elif orientation == 1:
                new_orientation = 2
            elif orientation == 2:
                new_orientation = 3
            else:
                new_orientation = 0

            return (x, y, new_orientation)

    def update_food_status(self):
        self.food_points = [0 for a in range(self.max_food_num)]

        enumFood = list(enumerate(self.food_positions))
        food_locations = [(food[0], food[1]) for idx, food in enumFood if self.food_alive_statuses[idx]]
        food_id = [idx for idx, food in enumFood if self.food_alive_statuses[idx]]

        player_locations = self.player_positions
        set_of_food_location = set(food_locations)

        cur_dist_to_food = [min([abs(px - fx) + abs(py - fy) for (fx, fy) in food_locations])
                            for (px, py) in self.player_positions]

        self.player_points = [0.01 * (prev_dist - cur_dist)
                              for prev_dist, cur_dist in zip(self.prev_dist_to_food, cur_dist_to_food)]
        self.prev_dist_to_food = cur_dist_to_food

        self.food_points = [0 for _ in range(self.max_food_num)]
        player_id_counter = 0
        for player_loc in player_locations:
            player_vicinities = [((player_loc[0] + a[0]), (player_loc[1] + a[1])) for a in
                                 [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            for player_vic in player_vicinities:
                if player_vic in set_of_food_location:
                    center = player_vic
                    enumerated = enumerate(player_locations)
                    close = [x for (x, (a, b)) in enumerated if abs(a - center[0]) + abs(b - center[1])
                             <= self.coopRadius]
                    if len(close) > 1:

                        self.player_points[player_id_counter] += self.groupMultiplier * len(close)
                        food_index = food_locations.index(center)
                        self.food_points[food_id[food_index]] += -1
                        self.food_alive_statuses[food_id[food_index]] = False
                        self.food_frozen_time[food_id[food_index]] = self.food_freeze_rate

                        self.grid[center[0]][center[1]] = 1
                        if "full_rgb" in self.obs_type:
                            self.RGB_grid[center[0]][center[1]] = [0, 0, 0]
                        self.RGB_padded_grid[center[0] + self.pads][center[1] + self.pads] \
                            = [0, 0, 0]

            player_id_counter += 1

        for idx, food in enumerate(self.food_positions):
            if self.food_alive_statuses[idx]:
                self.grid[self.food_positions[idx][0]][self.food_positions[idx][1]] = 3
                if "full_rgb" in self.obs_type:
                    self.RGB_grid[self.food_positions[idx][0]][self.food_positions[idx][1]] = [255, 0, 0]
                self.RGB_padded_grid[self.food_positions[idx][0] + self.pads][self.food_positions[idx][1] + self.pads] \
                    = [255, 0, 0]

    def update_state(self, hunter_collective_action, food_collective_action):
        self.remaining_timesteps -= 1
        self.update_status()
        self.revive()

        prev_player_position = self.player_positions
        prev_player_orientation = self.player_orientation
        prev_food_position = self.food_positions
        prev_food_orientation = self.food_orientation

        update_results_player = self.calculate_new_position(hunter_collective_action, prev_player_position,
                                                            prev_player_orientation)
        post_player_position = [(a, b) for (a, b, c) in update_results_player]
        post_player_orientation = [c for (a, b, c) in update_results_player]
        self.player_orientation = post_player_orientation

        update_results_food = self.calculate_new_position(food_collective_action, prev_food_position,
                                                          prev_food_orientation)
        post_food_position = [(a, b) for (a, b, c) in update_results_food]
        post_food_orientation = [c for (a, b, c) in update_results_food]
        self.food_orientation = post_food_orientation

        prev_positions = [None] * (len(prev_player_position) + len(prev_food_position))
        post_positions = [None] * (len(prev_player_position) + len(prev_food_position))
        types = [None] * (len(prev_player_position) + len(prev_food_position))
        food_status = [False] * (len(prev_player_position) + len(prev_food_position))

        for a in range(len(prev_player_position)):
            prev_positions[a] = prev_player_position[a]
            post_positions[a] = post_player_position[a]
            types[a] = "player"

        for a in range(len(prev_food_position)):
            prev_positions[a + len(prev_player_position)] = prev_food_position[a]
            post_positions[a + len(post_player_position)] = post_food_position[a]
            types[a + len(post_player_position)] = "food"
            if self.food_alive_statuses[a]:
                food_status[a + len(post_player_position)] = True

        # Calculate player intersection
        a, seen, result = post_positions, {}, {}
        for idx, item in enumerate(a):
            next_pass = True
            if types[idx] == "food" and not food_status[idx]:
                next_pass = False

            if next_pass:
                if item not in seen:
                    result[item] = [idx]
                    seen[item] = types[idx]
                else:
                    result[item].append(idx)

        groupings = list(result.values())
        doubles = [t for t in groupings if len(t) > 1]
        while len(doubles) > 0:
            res = set([item for sublist in doubles for item in sublist])
            for ii in range(len(post_positions)):
                if ii in res:
                    post_positions[ii] = prev_positions[ii]

            a, seen, result = post_positions, {}, {}
            for idx, item in enumerate(a):
                next_pass = True
                if types[idx] == "food" and not food_status[idx]:
                    next_pass = False

                if next_pass:
                    if item not in seen:
                        result[item] = [idx]
                        seen[item] = types[idx]
                    else:
                        result[item].append(idx)

            groupings = list(result.values())
            doubles = [t for t in groupings if len(t) > 1]

        for a in self.food_positions:
            self.grid[a[0]][a[1]] = 1
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
        self.food_positions = post_positions[len(post_player_position):]

        for idx, a in enumerate(self.food_positions):
            if self.food_alive_statuses[idx]:
                self.grid[a[0]][a[1]] = 3
                if "full_rgb" in self.obs_type:
                    self.RGB_grid[a[0]][a[1]] = [255, 0, 0]
                self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 0, 0]

        for a in self.player_positions:
            self.grid[a[0]][a[1]] = 1
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]
        self.player_positions = post_positions[:len(post_player_position)]

        for idx, a in enumerate(self.player_positions):
            self.grid[a[0]][a[1]] = 2
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [255, 255, 255]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 255, 255]

        self.update_food_status()

    def observation_computation(self, obs_type, agent_type="player", agent_id=0):

        if obs_type == "main_player":
            observations = []
            for player_locs in self.player_positions:
                observations.append(player_locs[0])
                observations.append(player_locs[1])
            food_locs = [x for a in self.food_positions for x in list(a)]
            observations.extend(food_locs)

            for orientation in self.food_orientation:
                or_vector = [0] * 4
                or_vector[orientation] = 1
                observations.extend(or_vector)

            if self.with_oppo_mod:
                other_acts = self.other_player_acts
                if self.other_player_acts is None:
                    other_acts = [-1] * (self.num_players-1)
                observations.extend(other_acts)

            return np.asarray(observations)

        elif obs_type == "centralized_dqn":
            obs_list = []
            for agent_id in range(self.num_players):
                observations = []
                player_locs = [self.player_positions[agent_id][0], self.player_positions[agent_id][1]]
                food_locs = [x for a in self.food_positions for x in list(a)]
                observations.extend(player_locs)
                observations.extend(food_locs)

                for orientation in self.food_orientation:
                    or_vector = [0] * 4
                    or_vector[orientation] = 1
                    observations.extend(or_vector)
                obs_list.append(np.asarray(observations))
            return [tuple(obs_list)]

        elif obs_type == "comp_processed":
            return [self.player_positions, self.player_orientation,
                    self.food_positions, self.food_orientation, self.food_alive_statuses,
                    self.possibleCoordinates]

        elif obs_type == "full_graph":
            orientation = self.player_orientation[agent_id]
            position = self.player_positions[agent_id]

            def orientation_to_one_hot(orientation):
                orientation_list = [0] * 4
                orientation_list[orientation] = 1
                return orientation_list

            own_pos = list(position)
            own_pos.extend(orientation_to_one_hot(orientation))
            position_list = [own_pos]
            enemy_pos_list = [self.food_positions[0][0], self.food_positions[0][1], self.food_positions[1][0],
                              self.food_positions[1][1]]

            for a in self.food_orientation:
                f_one_hot_orientation = orientation_to_one_hot(a)
                enemy_pos_list.extend(f_one_hot_orientation)


            for idx in range(len(self.player_orientation)):
                if idx != agent_id:
                    other_pos = list(self.player_positions[idx])
                    other_pos.extend(orientation_to_one_hot(self.player_orientation[idx]))
                    position_list.append(other_pos)

            return (position_list, enemy_pos_list)

        elif obs_type == "partial_obs":
            if agent_type == "player":
                orientation = self.player_orientation[agent_id]
                pos_0, pos_1 = self.player_positions[agent_id][0], self.player_positions[agent_id][1]
            else:
                orientation = self.food_orientation[agent_id]
                pos_0, pos_1 = self.food_positions[agent_id][0], self.food_positions[agent_id][1]

            pos_0 = pos_0 + self.pads
            pos_1 = pos_1 + self.pads
            obs_grid = np.asarray(self.RGB_padded_grid)

            if orientation == 0:
                partial_ob = obs_grid[pos_0 - self.sight_radius:pos_0 + 1,
                             pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]


            elif orientation == 1:
                partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
                             pos_1:pos_1 + self.sight_radius + 1]

                partial_ob = partial_ob.transpose((1, 0, 2))
                partial_ob = partial_ob[::-1]

            elif orientation == 2:
                partial_ob = obs_grid[pos_0:pos_0 + self.sight_radius + 1,
                             pos_1 - self.sight_sideways:pos_1 + self.sight_sideways + 1]
                partial_ob = np.fliplr(partial_ob)
                partial_ob = partial_ob[::-1]

            elif orientation == 3:
                partial_ob = obs_grid[pos_0 - self.sight_sideways:pos_0 + self.sight_sideways + 1,
                             pos_1 - self.sight_radius:pos_1 + 1]
                partial_ob = partial_ob.transpose((1, 0, 2))
                partial_ob = np.fliplr(partial_ob)

            return partial_ob

    def step(self, action):
        hunter_collective_action = [action]
        self.other_player_acts = [player.act(obs) for player, obs in zip(self.other_agents_list,
                                                                        self.other_player_obses)]
        hunter_collective_action.extend(self.other_player_acts)
        food_collective_action = [prey.act(obs, epsilon=0.1) for prey, obs in zip(self.food_list, self.food_obses)]
        self.update_state(hunter_collective_action, food_collective_action)
        player_returns = (tuple([self.observation_computation(obs_type, agent_id=id)
                                 for id, obs_type in enumerate(self.player_obs_type)]),
                          self.player_points, [self.remaining_timesteps == 0 for
                                               a in range(len(self.player_points))], [{} for _ in
                                                                                      range(len(self.player_points))])

        food_returns = ([self.observation_computation(obs_type, agent_type="food", agent_id=id)
                         for id, obs_type in enumerate(self.food_obs_type)],
                        self.food_points, [self.remaining_timesteps == 0
                                           for a in range(self.max_food_num)])

        self.food_obses = food_returns[0]
        self.other_player_obses = player_returns[0][1:]

        return player_returns[0][0], player_returns[1][0], player_returns[2][0], {}

    def render(self, mode='human', close=False):
        if self.visualizer is None:
            self.visualizer = Visualizer(self.grid, self.grid_height, self.grid_width)

        self.visualizer.grid = self.grid
        self.visualizer.render()
        self.visualizer.render()

    def sample_teammates(self, num_sampled):
        poss_teammates = [RandomAgent, GreedyPredatorAgent, GreedyProbabilisticAgent,
                          TeammateAwarePredator, DistilledCoopAStarAgent, GraphDQNAgent]

        return [poss_teammates[np.random.choice(len(poss_teammates), 1)[0]](idx+1) for idx in range(num_sampled)]

class Visualizer(object):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    WIDTH = 20
    HEIGHT = 20

    MARGIN = 0

    # Create a 2 dimensional array. A two dimensional
    # array is simply a list of lists.

    def __init__(self, grid, grid_height=20, grid_width=20):
        import pygame
        self.pygame = pygame
        self.grid_height, self.grid_width = grid_height, grid_width
        self.grid = grid
        self.WINDOW_SIZE = [self.grid_height * self.HEIGHT, self.grid_width * self.WIDTH]

        self.pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        self.pygame.display.set_caption("Wolfpack")
        self.clock = pygame.time.Clock()

    def render(self):
        done = False
        for event in self.pygame.event.get():  # User did something
            if event.type == self.pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        self.screen.fill(self.BLACK)

        for row in range(self.grid_height):
            for column in range(self.grid_width):
                color = self.BLUE
                if self.grid[row][column] == 1:
                    color = self.BLACK
                elif self.grid[row][column] == 2:
                    color = self.WHITE
                elif self.grid[row][column] == 3:
                    color = self.RED
                self.pygame.draw.rect(self.screen,
                                      color,
                                      [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
                                       (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                                       self.WIDTH,
                                       self.HEIGHT])

        self.clock.tick(60)
        self.pygame.display.flip()

        if done:
            self.pygame.quit()
