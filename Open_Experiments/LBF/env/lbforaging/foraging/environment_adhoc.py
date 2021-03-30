import logging
from collections import namedtuple, defaultdict
from enum import IntEnum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
from lbforaging.agents.A2CAgents import A2CAgent
from lbforaging.agents.heuristic_agent import H1, H2, H3, H4, H5, H6, H7, H8


class Action(IntEnum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

        self.active = False

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=False,
        effective_max_num_players=3,
        init_num_players=3,
        seed=1235,
        with_shuffle=True,
        gnn_input=False,
        with_openness=True,
        with_gnn_shuffle=False,
        collapsed=False
    ):
        self.logger = logging.getLogger(__name__)
        self.seed_val = seed
        self.seed(seed)
        self.players = [Player() for _ in range(players)]
        self.npc_controller_list = [None] * (players - 1)
        self.npc_type_name_list = [None] * (players - 1)
        self.other_players_obs = [None] * (players - 1)
        self.other_players_actions = [-1] * (players - 1)
        self.max_num_npc = players - 1
        self.prev_actions = [-1] * (players - 1)
        self.all_prev_actions = [-1] * (players)
        self.with_shuffle = with_shuffle
        self.effective_max_num_players = effective_max_num_players
        self.remaining_idxs = []
        self.with_openness = with_openness
        self.field = np.zeros(field_size, np.int32)
        self.init_num_players_w_reset = init_num_players

        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None
        self.pre_rew_storage = None
        self.gnn_input = gnn_input
        self.pre_agent_num = 0
        self.pre_agent_queue = []
        self.with_gnn_shuffle = with_gnn_shuffle
        self.collapsed = collapsed

        # np.random.seed(seed)

        self.agent_state_span = [0] * len(self.players)
        self.init_num_players = init_num_players
        self.agent_queue = []

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps
        self._normalize_reward = normalize_reward

    @property
    def action_space(self):
        return gym.spaces.Discrete(6)

    @property
    def observation_space_base(self):
        return self._get_observation_space()

    @property
    def observation_space(self):
        return self._get_observation_space_real()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        field_x = self.field.shape[1]
        field_y = self.field.shape[0]
        # field_size = field_x * field_y

        max_food = self.max_food
        max_food_level = self.max_player_level * len(self.players)

        min_obs = [-1, -1, 0] * max_food + [-1, -1, 0] * len(self.players) + [0]
        max_obs = [field_x, field_y, max_food_level] * max_food + [
            field_x,
            field_y,
            self.max_player_level,
        ] * len(self.players) + [1]

        return gym.spaces.Dict(
            {'all_information': gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)})

    def _get_observation_space_real(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        field_x = self.field.shape[1]
        field_y = self.field.shape[0]
        # field_size = field_x * field_y

        max_food = self.max_food
        max_food_level = self.max_player_level * len(self.players)

        if not self.gnn_input:
            min_obs = [-1, -1, 0] * max_food + [-1, -1, 0] * len(self.players)
            max_obs = [field_x, field_y, max_food_level] * max_food + [
                field_x,
                field_y,
                self.max_player_level,
            ] * len(self.players)

            return gym.spaces.Dict(
                {'all_information': gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)})
        min_food_obs = [-1, -1, 0] * max_food
        min_player_obs = [-1, -1, 0] * len(self.players)
        min_available_players = [-1] * len(self.players)
        min_num_players = [0]
        min_prev_action = [-1] * (len(self.players) - 1)

        max_food_obs = [field_x, field_y, max_food_level] * max_food
        max_player_obs = [
                             field_x,
                             field_y,
                             self.max_player_level,
                         ] * len(self.players)
        max_available_players = [1] * len(self.players)
        max_num_players = [len(self.players)]
        max_prev_action = [5] * (len(self.players) - 1)

        if not self.with_gnn_shuffle:
            return gym.spaces.Dict({
                'food_info': gym.spaces.Box(np.array(min_food_obs), np.array(max_food_obs), dtype=np.int32),
                'player_info': gym.spaces.Box(np.array(min_player_obs), np.array(max_player_obs), dtype=np.int32),
                'player_filter': gym.spaces.Box(np.array(min_available_players), np.array(max_available_players),
                                                dtype=np.int32),
                'num_player': gym.spaces.Box(np.array(min_num_players), np.array(max_num_players), dtype=np.int32),
                'prev_actions': gym.spaces.Box(np.array(min_prev_action), np.array(max_prev_action), dtype=np.int32)
            })

        if not self.collapsed:
            return gym.spaces.Dict({
                'food_info': gym.spaces.Box(np.array(min_food_obs), np.array(max_food_obs), dtype=np.int32),
                'player_info': gym.spaces.Box(np.array(min_player_obs), np.array(max_player_obs), dtype=np.int32),
                'player_filter': gym.spaces.Box(np.array(min_available_players), np.array(max_available_players),
                                                dtype=np.int32),
                'num_player': gym.spaces.Box(np.array(min_num_players), np.array(max_num_players), dtype=np.int32),
                'prev_actions': gym.spaces.Box(np.array(min_prev_action), np.array(max_prev_action), dtype=np.int32),
                'player_info_shuffled': gym.spaces.Box(np.array(min_player_obs), np.array(max_player_obs),
                                                       dtype=np.int32)
            })

        teammate_info_len = 4
        food_info_len = 3 * max_food
        prev_actions = 1

        return gym.spaces.Box(
            low=-np.inf, high=+np.inf,
            shape=(
                len(self.players), teammate_info_len + food_info_len + prev_actions,
            )
        )

    @classmethod
    def from_obs(cls, obs):

        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    @property
    def active_players(self):
        return [p for p in self.players if p.active]

    def add_agent(self, id):
        if self.players[id].active:
            self.logger.warning("Agent already active.")
            return

        attempts = 0
        player = self.players[id]

        while attempts < 1000:
            row = self.np_random.randint(0, self.rows - 1)
            col = self.np_random.randint(0, self.cols - 1)
            if self._is_empty_location(row, col):
                player.setup(
                    (row, col),
                    self.np_random.randint(1, self.max_player_level),
                    self.field_size,
                )
                self.players[id].active = True
                break
            attempts += 1
        self._gen_valid_moves()

    def remove_agent(self, id):
        self.players[id].active = False
        self.players[id].position = (-1, -1)
        self._gen_valid_moves()

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.active_players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                   max(row - distance, 0): min(row + distance + 1, self.rows),
                   max(col - distance, 0): min(col + distance + 1, self.cols),
                   ]

        return (
            self.field[
            max(row - distance, 0): min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
              row, max(col - distance, 0): min(col + distance + 1, self.cols)
              ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):

        return [
            player
            for player in self.active_players
            if abs(player.position[0] - row) == 1
               and player.position[1] == col
               or abs(player.position[1] - col) == 1
               and player.position[0] == row
        ]

    def spawn_food(self, max_food, max_level):

        food_count = 0
        attempts = 0
        min_level = max_level if self.force_coop else 1

        while food_count < max_food and attempts < 1000:
            attempts += 1
            row = self.np_random.randint(1, self.rows - 1)
            col = self.np_random.randint(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_level
                if min_level == max_level
                else self.np_random.randint(min_level, max_level)
            )
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):

        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, max_player_level):

        for player in self.players:
            if not player.active:
                continue

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows - 1)
                col = self.np_random.randint(0, self.cols - 1)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.np_random.randint(1, max_player_level),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        if not player.active:
            return None
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if a.active and (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                   and (
                       max(
                           self._transform_to_neighborhood(
                               player.position, self.sight, a.position
                           )
                       )
                   )
                   <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self, observations):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space_base['all_information'].shape)
            if observation is None:
                return obs
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[self.max_food * 3 + 3 * i] = -1
                obs[self.max_food * 3 + 3 * i + 1] = -1
                obs[self.max_food * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_food * 3 + 3 * i] = p.position[0]
                obs[self.max_food * 3 + 3 * i + 1] = p.position[1]
                obs[self.max_food * 3 + 3 * i + 2] = p.level
            obs[-1] = 1.0
            return obs

        def get_player_reward(observation, idx):
            if not self.pre_rew_storage is None:
                if observation is None and self.pre_rew_storage[idx] == 0.0:
                    return 0.0
                if observation is None and self.pre_rew_storage[idx] != 0.0:
                    return self.pre_rew_storage[idx]
                for p in observation.players:
                    if p.is_self:
                        return p.reward
            else:
                if observation is None:
                    return 0.0
                for p in observation.players:
                    if p.is_self:
                        return p.reward

        nobs = [make_obs_array(ob) if self.players[idx].active else make_obs_array(None)
                for idx, ob in enumerate(observations)]
        nreward = [get_player_reward(obs, idx) for idx, obs in enumerate(observations)]
        ndone = all([obs.game_over if obs else True for obs in observations])
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = [{} for obs in observations]

        # todo this?:
        # return nobs, nreward, ndone, ninfo
        # use this line to enable heuristic agents:
        return list(zip(observations, nobs)), nreward, ndone, ninfo

    def _make_gym_obs_returns(self, observations):
        def get_player_reward(observation):
            if observation is None:
                return 0.0
            for p in observation.players:
                if p.is_self:
                    return p.reward

        nreward = [get_player_reward(obs) for obs in observations]
        return nreward

    def agent_remove_selector(self):
        active_agents_id = [idx for idx, player in enumerate(self.players) if player.active]
        self.remaining_idxs = [-1] * len(self.players)
        for ag_idx, agent_id in enumerate(self.agent_queue):
            self.remaining_idxs[ag_idx] = 0

        if len(active_agents_id) - 1 > 0:
            max_removed = len(active_agents_id) - 2
            if not self.with_openness:
                max_removed = 0
            removed_agents_id = [
                id for id in active_agents_id
                if self.agent_state_span[id] < 0
            ]

            removed_agents_id = [id for id in removed_agents_id if id != 0]
            if len(removed_agents_id) > 0:
                if max_removed < 1:
                    removed_agents_id = []
                else:
                    removed_agents_id = self.np_random.choice(
                        removed_agents_id, min(len(removed_agents_id), max_removed), replace=False
                    ).tolist()

            for ag_idx, agent_id in enumerate(self.agent_queue):
                if not (agent_id in removed_agents_id):
                    self.remaining_idxs[ag_idx] = 1

            removed_agents_id.sort(reverse=True)
            for id in removed_agents_id:
                self.remove_agent(id)
                for idx, value in enumerate(self.agent_queue):
                    if value == id:
                        del self.agent_queue[idx]
                self.agent_state_span[id] = self.np_random.choice(range(15, 25), 1)[0]
                self.npc_type_name_list[id - 1] = None
                self.npc_controller_list[id - 1] = None

    def agent_add_selector(self):
        inactive_agents_id = [idx for idx, player in enumerate(self.players) if not player.active]
        if len(inactive_agents_id) > len(self.players) - self.effective_max_num_players:
            # Compute max avail to not violate the max active agent constraints
            max_chosen_inactive = len(inactive_agents_id) - (len(self.players) - self.effective_max_num_players)

            avail_agents_id = [
                id for id in inactive_agents_id
                if self.agent_state_span[id] < 0
            ]

            added_agents_id = avail_agents_id
            if len(avail_agents_id) > max_chosen_inactive:
                added_agents_id = self.np_random.choice(avail_agents_id, max_chosen_inactive, replace=False).tolist()

            for id in added_agents_id:
                agent_type_name, agent_type = self.agent_type_selector()
                self.add_agent(id)
                self.agent_queue.append(id)
                self.agent_state_span[id] = self.np_random.choice(range(10, 20), 1)[0]
                if "H" in agent_type_name:
                    agent_type = agent_type(self.players[id])
                self.npc_controller_list[id - 1] = agent_type
                self.npc_type_name_list[id - 1] = agent_type_name

    def agent_type_selector(self):
        agent_types = ["H8", "H7", "H6", "H5", "A2C0", "H1", "H2", "H3", "H4"]
        agent_type = self.np_random.choice(agent_types, 1)[0]

        if agent_type == "H1":
            return agent_type, H1
        elif agent_type == "H2":
            return agent_type, H2
        elif agent_type == "H3":
            return agent_type, H3
        elif agent_type == "H4":
            return agent_type, H4
        elif agent_type == "H5":
            return agent_type, H5
        elif agent_type == "H6":
            return agent_type, H6
        elif agent_type == "H7":
            return agent_type, H7
        elif agent_type == "H8":
            return agent_type, H8
        else:
            a2c_index = int(agent_type[-1])
            return agent_type, A2CAgent(a2c_index)

    def reset(self):
        if self.init_num_players_w_reset < 0:
            self.init_num_players = self.np_random.choice(range(2, 6), 1)[0]
            self.effective_max_num_players = self.init_num_players

        self.seed_val = self.seed_val + 123
        self.seed(self.seed_val)
        self.npc_controller_list = [None] * (len(self.players) - 1)
        self.npc_type_name_list = [None] * (len(self.players) - 1)
        self.other_players_obs = [None] * (len(self.players) - 1)
        self.other_players_actions = [-1] * (len(self.players) - 1)

        self.agent_queue = []
        rearranged_prev_actions = [self.prev_actions[k - 1] for k in self.pre_agent_queue]
        while len(rearranged_prev_actions) < len(self.players) - 1:
            rearranged_prev_actions.append(-1)
        rearranged_prev_actions = np.array(rearranged_prev_actions, np.int32)
        all_prev_actions = self.all_prev_actions
        self.prev_actions = [-1] * (len(self.players) - 1)
        self.all_prev_actions = [-1] * len(self.players)

        chosen_player_index = self.np_random.choice(len(self.players) - 1, self.init_num_players - 1, replace=False) + 1
        chosen_player_index_final = [0]
        chosen_player_index_final.extend(chosen_player_index.tolist())
        self.max_num_npc = len(self.players) - 1
        self.remaining_idxs = []

        self.prev_actions = [-1] * self.max_num_npc

        for idx, player in enumerate(self.players):
            self.remove_agent(idx)

        self.remaining_idxs = [-1] * len(self.players)
        for idx in range(self.pre_agent_num):
            self.remaining_idxs[idx] = 0

        for idx in chosen_player_index_final:
            self.players[idx].active = True
            self.agent_queue.append(idx)

        id_cooldown = 1
        for idx in chosen_player_index_final[1:]:
            agent_type, agent_controller = self.agent_type_selector()
            if "H" in agent_type:
                agent_controller = agent_controller(self.players[idx])
            self.npc_type_name_list[idx - 1] = agent_type
            self.npc_controller_list[idx - 1] = agent_controller

        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level)
        self.spawn_food(
            self.max_food, max_level=sum([player.level for player in self.active_players])
        )
        self.current_step = 0
        self.pre_rew_storage = None
        self._game_over = False
        self._gen_valid_moves()

        self.agent_state_span = [
            self.np_random.choice(range(10, 20), 1)[0] if player.active else self.np_random.choice(range(15, 25), 1)[0]
            for player in self.players
        ]

        observations = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ninfo = self._make_gym_obs(observations)
        self.other_players_obs = [None] * (len(self.players) - 1)
        idx = 0
        for nob, player_type_name in zip(nobs[1:], self.npc_type_name_list):
            if not (player_type_name is None):
                if "A2C" in player_type_name:
                    self.other_players_obs[idx] = nob[1]
                else:
                    self.other_players_obs[idx] = nob[0]
            idx += 1

        if self.with_shuffle:
            returned_obs = np.copy(nobs[0][1])
            non_zero_idxes = [0]
            non_zero_idxes.extend([idx + 1 for idx, k in enumerate(self.npc_type_name_list) if k is not None])
            num_agents = sum([k is not None for k in self.npc_type_name_list]) + 1

            for i in range(len(self.players)):
                returned_obs[self.max_food * 3 + 3 * i] = -1
                returned_obs[self.max_food * 3 + 3 * i + 1] = -1
                returned_obs[self.max_food * 3 + 3 * i + 2] = 0

            for i in range(num_agents):
                returned_obs[self.max_food * 3 + 3 * non_zero_idxes[i]: self.max_food * 3 + 3 * non_zero_idxes[i] + 3] = \
                    nobs[0][1][self.max_food * 3 + 3 * i:self.max_food * 3 + 3 * i + 3]
            return {"all_information": returned_obs[:-1]}

        if not self.gnn_input:
            return {"all_information": nobs[0][1][:-1]}

        player_obs = np.zeros(len(self.players) * 3, np.int32)
        for i in range(len(self.players)):
            player_obs[3 * i] = -1
            player_obs[3 * i + 1] = -1
            player_obs[3 * i + 2] = 0

        sorted_idx_list = self.agent_queue.copy()
        sorted_idx_list.sort()
        shuffled_idxes = [sorted_idx_list.index(element) for element in self.agent_queue]

        for idx_write in range(len(self.agent_queue)):
            player_obs[3 * idx_write: 3 * idx_write + 3] = \
                nobs[0][1][
                3 * self.max_food + 3 * shuffled_idxes[idx_write]:3 * self.max_food + 3 * shuffled_idxes[idx_write] + 3]

        if self.with_gnn_shuffle:
            player_obs_shuffled = np.zeros(len(self.players) * 3, np.int32)

            for i in range(len(self.players)):
                player_obs_shuffled[3 * i] = -1
                player_obs_shuffled[3 * i + 1] = -1
                player_obs_shuffled[3 * i + 2] = 0

            non_zero_idxes = [0]
            non_zero_idxes.extend([idx + 1 for idx, k in enumerate(self.npc_type_name_list) if k is not None])
            num_agents = sum([k is not None for k in self.npc_type_name_list]) + 1

            for i in range(num_agents):
                player_obs_shuffled[3 * non_zero_idxes[i]: 3 * non_zero_idxes[i] + 3] = \
                    nobs[0][1][self.max_food * 3 + 3 * i:self.max_food * 3 + 3 * i + 3]

        food_info_obs = nobs[0][1][:3 * self.max_food]
        player_info = player_obs
        remaining_info = np.array(self.remaining_idxs, np.int32)
        num_agents = np.array([len(chosen_player_index_final)], np.int32)

        if not self.with_gnn_shuffle:
            return {
                "food_info": food_info_obs,
                "player_info": player_info,
                'player_filter': remaining_info,
                'num_player': num_agents,
                'prev_actions': rearranged_prev_actions
            }

        if not self.collapsed:
            return {
                "food_info": food_info_obs,
                "player_info": player_info,
                'player_filter': remaining_info,
                'num_player': num_agents,
                'prev_actions': rearranged_prev_actions,
                'player_info_shuffled': player_obs_shuffled
            }

        collapsed_loc = np.reshape(player_obs_shuffled, [-1, 3])
        alive_indicator = np.ones([collapsed_loc.shape[0], 1])
        alive_indicator[collapsed_loc[:, 0] == -1, 0] = 0

        collapsed_food_inf = np.reshape(food_info_obs, [1, -1])
        collapsed_food_inf = np.repeat(collapsed_food_inf, collapsed_loc.shape[0], axis=0)

        collapsed_data = np.append(alive_indicator, collapsed_loc, axis=-1)
        invalids = -np.ones([sum(alive_indicator[:, 0] == 0), collapsed_food_inf.shape[-1]])
        collapsed_food_inf[alive_indicator[:, 0] == 0, :] = invalids
        collapsed_data = np.append(collapsed_data, collapsed_food_inf, axis=-1)

        prev_action_data = np.reshape(all_prev_actions, [-1, 1])
        collapsed_data = np.append(collapsed_data, prev_action_data, axis=-1)

        return collapsed_data

    def step(self, action):
        self.current_step += 1

        for p in self.players:
            p.reward = 0
        actions = [action]

        other_actions = []
        for agent, a_type, ob in zip(self.npc_controller_list, self.npc_type_name_list, self.other_players_obs):
            if not (agent is None) and not (ob is None):
                if "A2C" in a_type:
                    other_actions.append(agent.step(ob))
                else:
                    other_actions.append(int(agent._step(ob)))
            else:
                other_actions.append(None)
        self.prev_actions = [int(x) if not x is None else -1 for x in other_actions]
        actions.extend(other_actions)
        prev_actions_collapsed = [int(x) if not x is None else -1 for x in actions]
        self.all_prev_actions = [int(x) if not x is None else -1 for x in actions]
        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions) if p.active
        ]
        self.pre_agent_num = sum([player.active for player in self.players])
        self.pre_agent_queue = self.agent_queue.copy()[1:]

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.active_players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = food
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            self.field[frow, fcol] = 0

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.active_players:
            p.score += p.reward

        self.pre_rew_storage = [player.reward for player in self.players]

        if len(self.agent_queue) > 1:
            pre_remove_queue = self.agent_queue.copy()[1:]
        else:
            pre_remove_queue = self.agent_queue.copy()

        self.agent_remove_selector()
        self.agent_add_selector()

        self.agent_state_span = [k - 1 for k in self.agent_state_span]
        observations_post_remove = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ninfo = self._make_gym_obs(observations_post_remove)
        self.other_players_obs = [None] * (len(self.players) - 1)

        idx = 0
        for nob, player_type_name in zip(nobs[1:], self.npc_type_name_list):
            if not (player_type_name is None):
                if "A2C" in player_type_name:
                    self.other_players_obs[idx] = nob[1]
                else:
                    self.other_players_obs[idx] = nob[0]
            idx += 1

        if self.with_shuffle:
            returned_obs = np.copy(nobs[0][1])
            non_zero_idxes = [0]
            non_zero_idxes.extend([idx + 1 for idx, k in enumerate(self.npc_type_name_list) if k is not None])
            num_agents = sum([k is not None for k in self.npc_type_name_list]) + 1

            for i in range(len(self.players)):
                returned_obs[self.max_food * 3 + 3 * i] = -1
                returned_obs[self.max_food * 3 + 3 * i + 1] = -1
                returned_obs[self.max_food * 3 + 3 * i + 2] = 0

            for i in range(num_agents):
                returned_obs[self.max_food * 3 + 3 * non_zero_idxes[i]: self.max_food * 3 + 3 * non_zero_idxes[i] + 3] = \
                    nobs[0][1][self.max_food * 3 + 3 * i:self.max_food * 3 + 3 * i + 3]

            return {"all_information": returned_obs[:-1]}, nreward[0], ndone, ninfo[0]

        if not self.gnn_input:
            return {"all_information": nobs[0][1][:-1]}, nreward[0], ndone, ninfo[0]

        player_obs = np.zeros(len(self.players) * 3, np.int32)
        for i in range(len(self.players)):
            player_obs[3 * i] = -1
            player_obs[3 * i + 1] = -1
            player_obs[3 * i + 2] = 0

        sorted_idx_list = self.agent_queue.copy()
        sorted_idx_list.sort()
        shuffled_idxes = [sorted_idx_list.index(element) for element in self.agent_queue]

        for idx_write in range(len(self.agent_queue)):
            player_obs[3 * idx_write: 3 * idx_write + 3] = \
                nobs[0][1][
                3 * self.max_food + 3 * shuffled_idxes[idx_write]:3 * self.max_food + 3 * shuffled_idxes[idx_write] + 3]

        if self.with_gnn_shuffle:
            player_obs_shuffled = np.zeros(len(self.players) * 3, np.int32)

            for i in range(len(self.players)):
                player_obs_shuffled[3 * i] = -1
                player_obs_shuffled[3 * i + 1] = -1
                player_obs_shuffled[3 * i + 2] = 0

            non_zero_idxes = [0]
            non_zero_idxes.extend([idx + 1 for idx, k in enumerate(self.npc_type_name_list) if k is not None])
            num_agents = sum([k is not None for k in self.npc_type_name_list]) + 1

            for i in range(num_agents):
                player_obs_shuffled[3 * non_zero_idxes[i]: 3 * non_zero_idxes[i] + 3] = \
                    nobs[0][1][self.max_food * 3 + 3 * i:self.max_food * 3 + 3 * i + 3]

        food_info_obs = nobs[0][1][:3 * self.max_food]
        player_info = player_obs
        remaining_info = np.array(self.remaining_idxs, np.int32)
        num_agents = np.array([sum([player.active for player in self.players])], np.int32)

        rearranged_prev_actions = [self.prev_actions[k - 1] for k in pre_remove_queue]
        while len(rearranged_prev_actions) < len(self.players) - 1:
            rearranged_prev_actions.append(-1)
        rearranged_prev_actions = np.array(rearranged_prev_actions, np.int32)

        if not self.with_gnn_shuffle:
            return {
                       'food_info': food_info_obs,
                       'player_info': player_info,
                       'player_filter': remaining_info,
                       'num_player': num_agents,
                       'prev_actions': rearranged_prev_actions
                   }, nreward[0], ndone, ninfo[0]

        if not self.collapsed:
            return {
                       'food_info': food_info_obs,
                       'player_info': player_info,
                       'player_filter': remaining_info,
                       'num_player': num_agents,
                       'prev_actions': rearranged_prev_actions,
                       'player_info_shuffled': player_obs_shuffled
                   }, nreward[0], ndone, ninfo[0]

        collapsed_loc = np.reshape(player_obs_shuffled, [-1, 3])
        alive_indicator = np.ones([collapsed_loc.shape[0], 1])
        alive_indicator[collapsed_loc[:, 0] == -1, 0] = 0

        collapsed_food_inf = np.reshape(food_info_obs, [1, -1])
        collapsed_food_inf = np.repeat(collapsed_food_inf, collapsed_loc.shape[0], axis=0)

        collapsed_data = np.append(alive_indicator, collapsed_loc, axis=-1)
        invalids = -np.ones([sum(alive_indicator[:, 0] == 0), collapsed_food_inf.shape[-1]])
        collapsed_food_inf[alive_indicator[:, 0] == 0, :] = invalids
        collapsed_data = np.append(collapsed_data, collapsed_food_inf, axis=-1)

        prev_action_data = np.reshape(prev_actions_collapsed, [-1, 1])
        collapsed_data = np.append(collapsed_data, prev_action_data, axis=-1)

        return collapsed_data, nreward[0], ndone, ninfo[0]

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        self.viewer.render(self)
