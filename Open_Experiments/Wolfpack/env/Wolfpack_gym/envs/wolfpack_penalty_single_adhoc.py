import copy
import pickle as pkl
import numpy as np
import threading
import gym
from gym import spaces
import time
import sys, os

from Wolfpack_gym.envs.wolfpack_penalty_single_adhoc_assets.Agents import *
from numpy.random import RandomState

"""
Class that generates random gridworld with obstacles for Wolfpack. 
Implementation based on the cellular automata algorithm. Fiddle with
the deathLimit and birthLimit parameters to change the density of the
resulting obstacles.
"""


class Generator(object):
    def __init__(self, size, deathLimit=4, birthLimit=3, seed=0):
        self.x_size = size[0]
        self.y_size = size[1]
        self.booleanMap = [[False] * k for k in [self.x_size] * self.y_size]
        self.probStartAlive = 0.82;
        self.deathLimit = deathLimit
        self.birthLimit = birthLimit
        self.copy = None
        self.seed = seed
        self.prng = RandomState(seed)

    def initialiseMap(self):
        for x in range(self.x_size):
            for y in range(self.y_size):
                if self.prng.random.uniform() < self.probStartAlive:
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


class WolfpackPenaltySingleAdhoc(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_height=20, grid_width=20, num_players=5, max_food_num=2,
                 sight_sideways=8, sight_radius=8, max_time_steps=200,
                 coop_radius=1, groupMultiplier=2, food_freeze_rate=0,
                 add_rate=0.05, del_rate=0.05, seed=None, max_player_num=5,
                 implicit_max_player_num=3, with_shuffling=True,
                 obs_type="main_player", with_random_grid=False, random_grid_dir=None,
                 prey_with_gpu=False, with_oppo_mod=True, close_penalty=0.5, rgb_obs=False,
                 tile_obs=False, tile_size=5, rnn_with_gnn=False, collapsed=False):

        # Define width and height of grid world
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.obs_type = obs_type
        self.with_oppo_mod = with_oppo_mod
        self.num_players = num_players
        self.close_penalty = close_penalty
        self.max_player_num = max_player_num
        self.implicit_max_player_num = implicit_max_player_num
        self.with_shuffling = with_shuffling
        self.tile_obs = tile_obs
        self.tile_size = tile_size
        self.rnn_with_gnn = rnn_with_gnn
        self.collapsed = collapsed

        N_DISCRETE_ACTIONS = 5
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        teammate_location_box_high = []
        for idx in range(self.max_player_num):
            teammate_location_box_high.append(self.grid_height - 1)
            teammate_location_box_high.append(self.grid_width - 1)

        opponent_box_high = []
        for idx in range(6 * max_food_num):
            if idx % 6 == 0:
                opponent_box_high.append(self.grid_height - 1)
            elif idx % 6 == 1:
                opponent_box_high.append(self.grid_width - 1)
            else:
                opponent_box_high.append(1)

        teammate_location_box_low = [0] * len(teammate_location_box_high)
        opponent_box_low = [0] * len(opponent_box_high)

        if self.with_oppo_mod:
            oppo_mod_box_high = [N_DISCRETE_ACTIONS - 1] * (self.max_player_num - 1)
            oppo_mod_box_low = [-1] * (self.max_player_num - 1)

        agent_remaining_box_high = [1] * self.max_player_num
        agent_remaining_box_low = [-1] * self.max_player_num

        self.with_rgb = rgb_obs
        self.observation_space = spaces.Dict({
            'teammate_location': spaces.Box(high=np.asarray(teammate_location_box_high),
                                            low=np.asarray(teammate_location_box_low)),
            'opponent_info': spaces.Box(high=np.asarray(opponent_box_high),
                                        low=np.asarray(opponent_box_low)),
            'remaining_flags': spaces.Box(high=np.asarray(agent_remaining_box_high),
                                          low=np.asarray(agent_remaining_box_low)),
            'num_agents': spaces.Box(high=np.asarray([0]),
                                     low=np.asarray([self.max_player_num]))
        })

        if self.with_oppo_mod:
            self.observation_space = spaces.Dict({
                'teammate_location': spaces.Box(high=np.asarray(teammate_location_box_high),
                                                low=np.asarray(teammate_location_box_low)),
                'opponent_info': spaces.Box(high=np.asarray(opponent_box_high),
                                            low=np.asarray(opponent_box_low)),
                'remaining_flags': spaces.Box(high=np.asarray(agent_remaining_box_high),
                                              low=np.asarray(agent_remaining_box_low)),
                'oppo_actions': spaces.Box(high=np.asarray(oppo_mod_box_high),
                                           low=np.asarray(oppo_mod_box_low)),
                'num_agents': spaces.Box(high=np.asarray([0]),
                                         low=np.asarray([self.max_player_num]))
            })

        if self.rnn_with_gnn:
            self.observation_space = spaces.Dict({
                'teammate_location': spaces.Box(high=np.asarray(teammate_location_box_high),
                                                low=np.asarray(teammate_location_box_low)),
                'opponent_info': spaces.Box(high=np.asarray(opponent_box_high),
                                            low=np.asarray(opponent_box_low)),
                'remaining_flags': spaces.Box(high=np.asarray(agent_remaining_box_high),
                                              low=np.asarray(agent_remaining_box_low)),
                'oppo_actions': spaces.Box(high=np.asarray(oppo_mod_box_high),
                                           low=np.asarray(oppo_mod_box_low)),
                'num_agents': spaces.Box(high=np.asarray([0]),
                                         low=np.asarray([self.max_player_num])),
                'teammate_location_shuffled': spaces.Box(high=np.asarray(teammate_location_box_high),
                                                         low=np.asarray(teammate_location_box_low))
            })

        if self.collapsed:
            teammate_loc_len = 2
            oppo_info_len = 12

            self.observation_space=spaces.Box(
                low=-np.inf, high=+np.inf,
                shape=(
                    self.max_player_num, teammate_loc_len+oppo_info_len+2,
                )
            )

        if self.with_rgb:
            obs_dict = {}
            low_bound = 0.0
            up_bound = 1.0
            obs_dict['all_obs'] = spaces.Box(low=low_bound, high=up_bound, shape=(self.grid_height, self.grid_width, 3), dtype=np.float32)
            if self.with_oppo_mod:
                obs_dict['oppo_actions'] = spaces.Box(high=np.asarray(oppo_mod_box_high), low=np.asarray(oppo_mod_box_low))

            self.observation_space = spaces.Dict(obs_dict)

        if self.tile_obs:
            obs_dict = {}
            low_bound = 0.0

            up_bound_agent = self.max_player_num
            up_bound_prey = max_food_num
            tile_nums = (self.grid_width-self.tile_size+1) * (self.grid_height-self.tile_size+1)

            # Low bound and high bound computation
            low_bound_arr = np.asarray([low_bound]*(2*tile_nums))
            high_bound_arr = np.asarray([up_bound_agent, up_bound_prey]*(tile_nums))

            obs_dict['all_obs'] = spaces.Box(low=low_bound_arr, high=high_bound_arr, dtype=np.float32)
            if self.with_oppo_mod:
                obs_dict['oppo_actions'] = spaces.Box(high=np.asarray(oppo_mod_box_high),
                                                      low=np.asarray(oppo_mod_box_low))

            self.observation_space = spaces.Dict(obs_dict)


        self.add_rate = add_rate
        self.del_rate = del_rate
        self.masking = []

        # Define seeds for the agent initial position generator
        if seed is None:
            seed = int(time.time())
        self.seed = seed
        # self.randomizer = random
        self.prng = RandomState(self.seed)
        self.scheduler = OpenScheduler(self.num_players - 1, self.add_rate, self.del_rate, self.implicit_max_player_num - 1, seed=self.seed)

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
        self.init_num_players = num_players
        self.valid_indices = [-1] * self.max_player_num
        for idx in range(self.init_num_players):
            self.valid_indices[idx] = idx
        self.prev_valid_indices = None
        self.max_food_num = max_food_num

        # Initialize maximum episode length and prey cooldown period
        self.max_time_steps = max_time_steps
        self.food_freeze_rate = food_freeze_rate

        self.other_player_acts = None
        self.other_player_obses = None
        self.all_player_acts = None

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
        self.seed = self.seed + 1250
        self.prng = RandomState(self.seed)
        self.visualizer = None
        self.num_players = self.init_num_players

        self.valid_indices = [-1] * self.max_player_num
        for idx in range(self.init_num_players):
            self.valid_indices[idx] = idx

        self.prev_valid_indices = None

        self.scheduler = OpenScheduler(self.num_players - 1, self.add_rate, self.del_rate, self.implicit_max_player_num - 1, seed=self.seed)

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
        player_loc_idx = self.prng.choice(range(len(self.possibleCoordinates)), self.num_players, replace=False).tolist()
        # player_loc_idx = random.sample(range(len(self.possibleCoordinates)), self.num_players)
        self.player_positions = [self.possibleCoordinates[a] for a in player_loc_idx]

        # Reset initial player orientation and points
        self.player_orientation = [0 for a in range(self.num_players)]
        self.player_points = [0 for a in range(self.num_players)]

        self.other_player_acts = None
        self.all_player_acts = None

        coordinates_no_player = [a for a in self.possibleCoordinates if a not in self.player_positions]
        food_loc_idx = self.prng.choice(range(len(coordinates_no_player)), self.max_food_num, replace=False).tolist()
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
        self.other_agents_list = self.sample_teammates(self.init_num_players - 1)

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

        remaining_flags = np.zeros(self.max_player_num) - 1
        for idx in range(len(self.other_agents_list) + 1):
            remaining_flags[idx] = 1

        if not self.collapsed:
            player_obses[0]["remaining_flags"] = remaining_flags
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
                idx = self.prng.choice(range(len(coordinates_no_player)), 1).tolist()[0]
                # idx = random.sample(range(len(coordinates_no_player)), 1)[0]
                while idx in idxes:
                    idx = self.prng.choice(range(len(coordinates_no_player)), 1).tolist()[0]
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

                    else:
                        self.player_points[player_id_counter] -= self.close_penalty

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
            observations = {}
            if self.tile_obs:
                mult1_list = []
                mult2_list = []

                for player_locs in self.player_positions:
                    mult1 = np.zeros([self.grid_height - self.tile_size + 1, 1])
                    for idx in range(self.grid_height-self.tile_size+1):
                        if idx + self.tile_size > player_locs[0] and idx <= player_locs[0]:
                            mult1[idx][0] = 1.0
                    mult1_list.append(mult1)

                for player_locs in self.player_positions:
                    mult2 = np.zeros([1, self.grid_width - self.tile_size + 1])
                    for idx in range(self.grid_height - self.tile_size + 1):
                        if idx + self.tile_size > player_locs[1] and idx <= player_locs[1]:
                            mult2[0][idx] = 1.0
                    mult2_list.append(mult2)

                player_mat = np.reshape(sum([a.dot(b) for a,b in zip(mult1_list, mult2_list)]), -1)

                mult1_list = []
                mult2_list = []

                for food_locs in self.food_positions:
                    mult1 = np.zeros([self.grid_height - self.tile_size + 1, 1])
                    for idx in range(self.grid_height - self.tile_size + 1):
                        if idx + self.tile_size > food_locs[0] and idx <= food_locs[0]:
                            mult1[idx][0] = 1.0
                    mult1_list.append(mult1)

                for food_locs in self.food_positions:
                    mult2 = np.zeros([1, self.grid_width - self.tile_size + 1])
                    for idx in range(self.grid_height - self.tile_size + 1):
                        if idx + self.tile_size > food_locs[1] and idx <= food_locs[1]:
                            mult2[0][idx] = 1.0
                    mult2_list.append(mult2)

                food_mat = np.reshape(sum([a.dot(b) for a, b in zip(mult1_list, mult2_list)]), -1)
                observations["all_obs"] = np.concatenate((player_mat, food_mat), axis=None)

                if self.with_oppo_mod:
                    oppo_actions = []
                    other_acts = self.other_player_acts
                    if self.other_player_acts is None:
                        other_acts = [-1] * (self.max_player_num - 1)
                    while len(other_acts) < self.max_player_num - 1:
                        other_acts.append(-1)
                    oppo_actions.extend(other_acts)
                    observations['oppo_actions'] = oppo_actions

                return observations

            if not self.with_rgb:
                if self.with_shuffling:
                    teammate_locations = [-1] * (2*self.max_player_num)
                    for idx, player_locs in enumerate(self.player_positions):
                        exact_loc = [idx_s for idx_s, idx_st in enumerate(self.valid_indices) if idx_st==idx][0]
                        teammate_locations[2*exact_loc] = player_locs[0]
                        teammate_locations[2*exact_loc+1] = player_locs[1]

                else:
                    teammate_locations = []
                    for player_locs in self.player_positions:
                        teammate_locations.extend(list(player_locs))
                    while len(teammate_locations) < (2*self.max_player_num):
                        teammate_locations.append(-1)

                if self.rnn_with_gnn:
                    teammate_locations_shuffled = [-1] * (2 * self.max_player_num)
                    for idx, player_locs in enumerate(self.player_positions):
                        exact_loc = [idx_s for idx_s, idx_st in enumerate(self.valid_indices) if idx_st == idx][0]
                        teammate_locations_shuffled[2 * exact_loc] = player_locs[0]
                        teammate_locations_shuffled[2 * exact_loc + 1] = player_locs[1]

                    teammate_locations = []
                    observations['teammate_location_shuffled'] = teammate_locations_shuffled
                    for player_locs in self.player_positions:
                        teammate_locations.extend(list(player_locs))
                    while len(teammate_locations) < (2 * self.max_player_num):
                        teammate_locations.append(-1)

                food_locs = [x for a in self.food_positions for x in list(a)]
                opponent_info = []
                opponent_info.extend(food_locs)

                for orientation in self.food_orientation:
                    or_vector = [0] * 4
                    or_vector[orientation] = 1
                    opponent_info.extend(or_vector)

                observations['teammate_location'] = np.asarray(teammate_locations)
                observations['opponent_info'] = np.asarray(opponent_info)

                if self.collapsed:
                    collapsed_loc = np.reshape(observations['teammate_location_shuffled'], [-1,2])
                    alive_indicator = np.ones([collapsed_loc.shape[0],1])
                    alive_indicator[collapsed_loc[:,0] == -1,0] = 0
                    collapsed_oppo_inf = np.reshape(observations['opponent_info'], [1,-1])
                    collapsed_oppo_inf = np.repeat(collapsed_oppo_inf, collapsed_loc.shape[0], axis=0)

                    collapsed_data = np.append(alive_indicator, collapsed_loc, axis=-1)
                    invalids = -np.ones([sum(alive_indicator[:,0]==0), collapsed_oppo_inf.shape[-1]])
                    collapsed_oppo_inf[alive_indicator[:,0]==0,:] = invalids
                    collapsed_data = np.append(collapsed_data, collapsed_oppo_inf, axis=-1)

            else:
                observation_matrix = np.zeros([self.grid_height, self.grid_width, 3])
                if len(self.player_positions) > 1 :
                    for pos in self.player_positions[1:]:
                        observation_matrix[pos[0]][pos[1]] = np.asarray([1.0, 0.0, 0.0])
                observation_matrix[self.player_positions[0][0]][self.player_positions[0][1]] = np.asarray([0.0, 1.0, 0.0]) 
                for pos in self.food_positions:
                    observation_matrix[pos[0]][pos[1]] = np.asarray([0.0, 0.0, 1.0])

                observations['all_obs'] = observation_matrix

            if self.with_oppo_mod:
                oppo_actions = []
                other_acts = self.other_player_acts
                if self.other_player_acts is None:
                    other_acts = [-1] * (self.max_player_num - 1)
                while len(other_acts) < self.max_player_num - 1:
                    other_acts.append(-1)

                if self.collapsed:
                    all_acts_shuffled = [-1] * (self.max_player_num)
                    if not self.all_player_acts is None:
                        for idx, prev_acts in enumerate(self.all_player_acts):
                            exact_loc = [idx_s for idx_s, idx_st in enumerate(self.prev_valid_indices) if idx_st == idx][0]
                            all_acts_shuffled[exact_loc] = prev_acts

                    all_acts = np.reshape(np.asarray(all_acts_shuffled), [-1,1])
                    collapsed_data = np.append(collapsed_data, all_acts, axis=-1)
                    observations = collapsed_data

                oppo_actions.extend(other_acts)
                if not self.collapsed:
                    observations['oppo_actions'] = oppo_actions

            if not self.with_rgb:
                if not self.collapsed:
                    observations['num_agents'] = np.asarray([self.num_players])
            return observations

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
        self.all_player_acts = hunter_collective_action
        food_collective_action = [prey.act(obs, epsilon=0.1) for prey, obs in zip(self.food_list, self.food_obses)]
        self.update_state(hunter_collective_action, food_collective_action)

        deleted, agents, new_types = self.scheduler.open_process()
        self.masking = [idx for idx in range(len(self.other_agents_list) + 1) if not idx in deleted]
        reverse_mask = [idx for idx in range(len(self.other_agents_list) + 1) if idx in deleted]
        self.prev_valid_indices = self.valid_indices.copy()
        self.del_agent(deleted)
        self.add_agent(agents, new_types)
        for idx, agent in enumerate(self.other_agents_list):
            agent.agent_id = idx + 1

        player_returns = (tuple([self.observation_computation(obs_type, agent_id=id)
                                 for id, obs_type in enumerate(self.player_obs_type)]),
                          self.player_points, [self.remaining_timesteps == 0 for
                                               a in range(len(self.player_points))], [{} for _ in
                                                                                      range(len(self.player_points))])

        remaining_flags = np.zeros(self.max_player_num) - 1
        for idx in self.masking:
            remaining_flags[idx] = 1

        for agent in self.other_agents_list:
            if "GraphDQNAgent" in str(type(agent)):
                agent.remaining_indices = remaining_flags

        remaining_flags = np.zeros(self.max_player_num) - 1
        for idx in self.masking:
            remaining_flags[idx] = 1
        for idx in reverse_mask:
            remaining_flags[idx] = 0
        if not self.collapsed:
            player_returns[0][0]["remaining_flags"] = remaining_flags

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
                          TeammateAwarePredator, DistilledCoopAStarAgent, GraphDQNAgent, 
                          GreedyWaitingAgent, GreedyProbabilisticWaitingAgent, 
                          TeammateAwareWaitingAgent]
        samples = [poss_teammates[self.prng.choice(range(len(poss_teammates)), 1)[0]](idx+1) for idx in range(num_sampled)]
        return samples

    def add_agent(self, new_agent, new_types):
        def_orientation = 0
        available_pos = list(set(self.possibleCoordinates) - set(self.player_positions) -
                                 set(self.food_positions))
        pos_idxes = self.prng.choice(len(available_pos), len(new_agent), replace=False).tolist()
        added_pos = [available_pos[a] for a in pos_idxes]
        orientation = [def_orientation for _ in range(len(added_pos))]

        self.player_orientation.extend(orientation)
        self.player_positions.extend(added_pos)
        self.player_obs_type.extend(new_types)
        self.other_agents_list.extend(new_agent)
        for a in added_pos:
            self.grid[a[0]][a[1]] = 2
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [255, 255, 255]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [255, 255, 255]
            self.num_players += 1

            offset = max(self.valid_indices) + 1
            possible_indices = [idx_s for idx_s, a in enumerate(self.valid_indices) if a == -1]
            idx_val = self.prng.choice(len(possible_indices), 1).tolist()[0]
            self.valid_indices[possible_indices[idx_val]] = offset

    def del_agent(self, agent_id):
        for idx in agent_id:
            a = self.player_positions[idx]
            self.grid[a[0]][a[1]] = 1
            if "full_rgb" in self.obs_type:
                self.RGB_grid[a[0]][a[1]] = [0, 0, 0]
            self.RGB_padded_grid[a[0] + self.pads][a[1] + self.pads] = [0, 0, 0]

            self.player_positions.pop(idx)
            self.other_agents_list.pop(idx - 1)
            self.player_orientation.pop(idx)
            self.player_obs_type.pop(idx)
            self.num_players -= 1

            for idx_val, a in enumerate(self.valid_indices):
                if a > idx:
                    self.valid_indices[idx_val] -= 1
                if a == idx:
                    self.valid_indices[idx_val] = -1


class OpenScheduler(object):
    def __init__(self, num_agents, add_rate, remove_rate, max_available_agents, min_alive_time=25, seed=0):
        self.available_agents = num_agents
        self.max_available_agents = max_available_agents
        self.alive_time = [0] * self.available_agents
        self.min_alive_time = min_alive_time
        self.seed = seed
        self.prng = RandomState(self.seed)
        self.timestep = 0

        # Use geometric distribution to sample remove or del
        self.geometric_add_rate = add_rate
        self.geometric_remove_rate = remove_rate

    def add_agents(self, agent_nums):
        new_agent = []
        new_obs_type = []
        for _ in range(agent_nums):
            agent_class = self.agent_type_sampler()
            agent = agent_class(self.available_agents + 1)
            new_agent.append(agent)
            new_obs_type.append(agent.obs_type)
            self.available_agents += 1
            self.alive_time.append(0)
        return new_agent, new_obs_type

    def del_agents(self, agent_idxs):
        agent_idxs_sorted = agent_idxs.copy()
        agent_idxs_sorted.sort(reverse=True)

        for idxes in agent_idxs_sorted:
            del self.alive_time[idxes-1]
        self.available_agents -= len(agent_idxs_sorted)
        return agent_idxs_sorted

    def agent_type_sampler(self):
        all_agent_types = [RandomAgent, GreedyPredatorAgent, GreedyProbabilisticAgent,
                          TeammateAwarePredator, DistilledCoopAStarAgent, GraphDQNAgent,
                          GreedyWaitingAgent, GreedyProbabilisticWaitingAgent,
                          TeammateAwareWaitingAgent]

        agent_id_type = self.prng.choice(len(all_agent_types),1).tolist()[0]
        return all_agent_types[agent_id_type]

    def agent_removal_sampler(self):
        eligible_idxs = [idx for idx, alive_dur in enumerate(self.alive_time) if alive_dur > self.min_alive_time]
        removed_amount = min(min(self.prng.choice(2, 1, p=[0.7, 0.3])[0] + 1, self.available_agents-1),
                             len(eligible_idxs))
        removed_indices = []
        if not removed_amount == 0:
            removed_indices = self.prng.choice(len(eligible_idxs), removed_amount, replace=False).tolist()
            removed_indices = [eligible_idxs[k] + 1 for k in removed_indices]
        return removed_indices

    def open_process(self):
        self.alive_time = [x+1 for x in self.alive_time]
        self.timestep+=1
        remove = False
        if self.prng.uniform() < self.geometric_remove_rate:
            remove = True

        add = False
        if self.prng.uniform() < self.geometric_add_rate:
            add = True

        deleted_idxs = []
        if remove:
            removed_indices = self.agent_removal_sampler()
            deleted_idxs = self.del_agents(removed_indices)

        new_agents, new_obs_type = [], []
        if add:
            agent_nums = min(self.prng.choice(2, 1, p=[0.7, 0.3])[0] + 1,
                                self.max_available_agents - self.available_agents)
            new_agents, new_obs_type = self.add_agents(agent_nums)
        return deleted_idxs, new_agents, new_obs_type

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
