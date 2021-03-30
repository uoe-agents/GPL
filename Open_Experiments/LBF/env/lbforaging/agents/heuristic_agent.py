import random
import numpy as np
from lbforaging.foraging.agent import Agent
from enum import IntEnum

class Action(IntEnum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5

class HeuristicAgent(Agent):
    name = "Heuristic Agent"

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        return np.rint(coords.mean(axis=0))

    def _furthest_player(self, players, pos):
        coords = [player.position for player in players]
        if len(coords) != 0:
            dists = [(loc[0]-pos[0])**2 + (loc[1]-pos[1])**2 for loc in coords]
            max_dists = max(dists)
            idx_max = np.random.choice([idx for idx, dist in enumerate(dists) if max_dists == dist], 1)[0]

            return coords[idx_max]

        return None

    def _eligible_furthest_player(self, players, level, pos):
        coords = [player.position for player in players]
        levels = [player.level for player in players]

        leader_id = -1
        max_level = float("-inf")
        for idx, lev in enumerate(levels):
            if lev > level and lev > max_level:
                max_level = lev
                leader_id = idx

        if leader_id != -1:
            return coords[leader_id]
        else:
            return self._furthest_player(players, pos)


    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        if r < y and Action.NORTH in allowed:
            return Action.NORTH
        elif r > y and Action.SOUTH in allowed:
            return Action.SOUTH
        elif c > x and Action.EAST in allowed:
            return Action.EAST
        elif c < x and Action.WEST in allowed:
            return Action.WEST
        else:
            raise ValueError("No simple path found")

    def step(self, obs):
        raise NotImplemented("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
    """
	H1 agent always goes to the closest food
	"""

    name = "H1"

    def step(self, obs):
        try:
            r, c = self._closest_food(obs)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H2(HeuristicAgent):
    """
	H2 Agent goes to the one visible food which is closest to the centre of visible players
	"""

    name = "H2"

    def step(self, obs):

        players_center = self._center_of_players(obs.players)

        try:
            r, c = self._closest_food(obs, None, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H3(HeuristicAgent):
    """
	H3 Agent always goes to the closest food with compatible level
	"""

    name = "H3"

    def step(self, obs):

        try:
            r, c = self._closest_food(obs, self.level)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H4(HeuristicAgent):
    """
	H4 Agent goes to the one visible food which is closest to all visible players
	 such that the sum of their and H4's level is sufficient to load the food
	"""

    name = "H4"

    def step(self, obs):

        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])

        try:
            r, c = self._closest_food(obs, players_sum_level, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)

class H5(HeuristicAgent):
    """
    Agent goes to farthest food in the grid.
    """
    name = "H5"
    def step(self, obs):
        try:
            r, c = self._farthest_food(obs)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position
        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)

class H6(HeuristicAgent):
    """
    Agent goes to farthest food in the grid.
    """
    name = "H6"
    def step(self, obs):
        try:
            r, c = self._highest_eligible_food(obs, self.level)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H7(HeuristicAgent):
    """
    Agent goes to farthest food according to agent furthest from the agent.
    """
    name = "H7"
    def step(self, obs):
        y, x = self.observed_position
        farthest_player = self._furthest_player(obs.players, self.observed_position)
        if farthest_player is None:
            return random.choice(obs.actions)

        if not (obs.field > 0).any():
            try:
                return self._move_towards(farthest_player, obs.actions)
            except ValueError:
                return random.choice(obs.actions)

        try:
            r, c = self._farthest_food(obs, start=farthest_player)
        except TypeError:
            return random.choice(obs.actions)

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)

class H8(HeuristicAgent):
    """
    Agent goes to farthest food according to agent furthest from the agent.
    """
    name = "H8"
    def step(self, obs):
        y, x = self.observed_position
        leader_loc = self._eligible_furthest_player(obs.players, self.level, self.observed_position)
        if leader_loc is None:
            return random.choice(obs.actions)

        if not (obs.field > 0).any():
            try:
                return self._move_towards(leader_loc, obs.actions)
            except ValueError:
                return random.choice(obs.actions)

        try:
            leader_level = 0
            for player in obs.players:
                if player.position == leader_loc:
                    leader_level = player.level
            r, c = self._highest_eligible_food(obs, leader_level)
        except TypeError:
            return random.choice(obs.actions)

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)