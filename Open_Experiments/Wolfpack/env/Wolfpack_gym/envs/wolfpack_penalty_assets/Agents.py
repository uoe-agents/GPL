import random
import math
import queue as Q

from Wolfpack_gym.envs.wolfpack_penalty_assets.ReplayMemory import ReplayMemoryLite
from Wolfpack_gym.envs.wolfpack_penalty_assets.QNetwork import DQN
from Wolfpack_gym.envs.wolfpack_penalty_assets.misc import hard_copy, soft_copy
import torch
import torch.optim as optim
import numpy as np

# import ray

class Agent(object):
    def __init__(self, agent_id, obs_type):
        self.agent_id = agent_id
        self.obs_type = obs_type

    def get_obstype(self):
        return self.obs_type

class RandomAgent(Agent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(RandomAgent, self).__init__(agent_id, obs_type)
        self.color = (148, 0, 211)

    def act(self, obs=None):
        return random.randint(0, 4)


class GreedyPredatorAgent(Agent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(GreedyPredatorAgent, self).__init__(agent_id, obs_type)
        self.color = (51, 255, 51)

    def act(self, obs):
        agent_pos = obs[0][self.agent_id]
        agent_orientation = obs[1][self.agent_id]
        oppo_pos = obs[2]
        oppo_alive_stats = obs[4]
        poss_locs = obs[5]

        pounce_res = self.pounce_prey(agent_pos, agent_orientation, oppo_pos, oppo_alive_stats)
        if pounce_res != -1:
            return pounce_res
        approach_res = self.approach_prey(agent_pos, agent_orientation, oppo_pos, oppo_alive_stats, poss_locs)
        if approach_res != -1:
            return approach_res
        else:
            return random.randint(0, 4)

    def pounce_prey(self, agent_pos, agent_orientation, oppo_pos, oppo_alive_stats):
        next_to_oppo = [self.computeManhattanDistance(agent_pos, oppo_next) < 2 for
                        idx, oppo_next in enumerate(oppo_pos) if oppo_alive_stats[idx]]
        next_to_oppo_idx = [idx for
                            idx, oppo_next in enumerate(oppo_pos) if oppo_alive_stats[idx]]

        if not any(next_to_oppo):
            return -1
        else:
            point1 = agent_pos
            point2 = oppo_pos[next_to_oppo_idx[next_to_oppo.index(True)]]
            y_del = point1[0] - point2[0]
            x_del = point1[1] - point2[1]

            compass_dir = 0
            if x_del > 0:
                compass_dir = 3
            elif x_del < 0:
                compass_dir = 1
            elif y_del < 0:
                compass_dir = 2

            real_dir = (compass_dir - agent_orientation) % 4
            return real_dir

    def approach_prey(self, agent_pos, agent_orientation, oppo_pos, oppo_alive_stats, possible_positions):
        adders = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        boundary_points = [(pos[0] + adder[0], pos[1] + adder[1])
                           for (idx, pos) in enumerate(oppo_pos)
                           for adder in adders if oppo_alive_stats[idx]
                           if (pos[0] + adder[0], pos[1] + adder[1]) in possible_positions]

        if len(boundary_points) != 0:
            manhattan_distances = [self.computeManhattanDistance(agent_pos, boundary_point)
                                   for boundary_point in boundary_points]
            min_idx = manhattan_distances.index(min(manhattan_distances))
            point1 = agent_pos
            point2 = boundary_points[min_idx]

            y_del = point1[0] - point2[0]
            x_del = point1[1] - point2[1]

            next_point_x = [point1[0], point1[1]]
            next_point_y = [point1[0], point1[1]]

            compass_dir_y = 0
            compass_dir_x = 1

            if y_del < 0:
                compass_dir_y = 2

            if x_del > 0:
                compass_dir_x = 3

            if compass_dir_y == 0:
                next_point_y[0] = next_point_y[0] - 1
            else:
                next_point_y[0] = next_point_y[0] + 1

            if compass_dir_x == 1:
                next_point_x[1] = next_point_x[1] + 1
            else:
                next_point_x[1] = next_point_x[1] - 1

            available_flags = [tuple(next_point_y) in possible_positions, tuple(next_point_x) in possible_positions]

            if not any(available_flags):
                return random.randint(0, 3)
            elif all(available_flags):
                if abs(y_del) > abs(x_del):
                    return (compass_dir_y - agent_orientation) % 4
                else:
                    return (compass_dir_x - agent_orientation) % 4
            else:
                if available_flags[0]:
                    return (compass_dir_y - agent_orientation) % 4

            return (compass_dir_x - agent_orientation) % 4
        return -1

    def computeManhattanDistance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


class GreedyWaitingAgent(GreedyPredatorAgent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(GreedyWaitingAgent, self).__init__(agent_id, obs_type)
        self.waiting_radius = np.random.choice(3, 1, p=[0.4, 0.4, 0.2]) + 2
        self.group_num_threshold = np.random.choice(2, 1, p=[0.7, 0.3]) + 1

    def act(self, obs):
        agent_pos = obs[0][self.agent_id]
        agent_orientation = obs[1][self.agent_id]
        oppo_pos = obs[2]
        oppo_alive_stats = obs[4]
        poss_locs = obs[5]

        in_proximity_alone = self.in_proximity_alone(agent_pos, obs[0],
                                                     agent_orientation, oppo_pos,
                                                     oppo_alive_stats, poss_locs)

        if in_proximity_alone != -1:
            return in_proximity_alone
        else:
            return super(GreedyWaitingAgent, self).act(obs)

    def in_proximity_alone(self, agent_pos, team_pos, agent_orientation, oppo_pos, oppo_alive_stats,
                           possible_positions):

        adders = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        boundary_points = [(idx, (pos[0] + adder[0], pos[1] + adder[1]))
                           for (idx, pos) in enumerate(oppo_pos)
                           for adder in adders if oppo_alive_stats[idx]
                           if (pos[0] + adder[0], pos[1] + adder[1])
                           in possible_positions]
        manhattan_distances = [super(GreedyWaitingAgent, self).computeManhattanDistance(agent_pos,
                                boundary_point[1])
                               for boundary_point in boundary_points]

        if len(manhattan_distances) != 0 :
            min_idx = manhattan_distances.index(min(manhattan_distances))
            closest_agent_loc = oppo_pos[boundary_points[min_idx][0]]

            if super(GreedyWaitingAgent, self).computeManhattanDistance(agent_pos, closest_agent_loc) <= self.waiting_radius:
                waiting_teammates = 0
                for loc in team_pos:
                    if loc != agent_pos and super(GreedyWaitingAgent, self).computeManhattanDistance(loc,
                                                                     closest_agent_loc) <=  self.waiting_radius:
                        waiting_teammates += 1

                if waiting_teammates < self.group_num_threshold:
                    point1 = agent_pos
                    point2 = closest_agent_loc

                    y_del = point1[0] - point2[0]
                    x_del = point1[1] - point2[1]

                    next_point_x = [point1[0], point1[1]]
                    next_point_y = [point1[0], point1[1]]

                    compass_dir_y = 2
                    compass_dir_x = 3

                    if y_del < 0:
                        compass_dir_y = 0

                    if x_del > 0:
                        compass_dir_x = 1

                    if compass_dir_y == 0:
                        next_point_y[0] = next_point_y[0] - 1
                    else:
                        next_point_y[0] = next_point_y[0] + 1

                    if compass_dir_x == 1:
                        next_point_x[1] = next_point_x[1] + 1
                    else:
                        next_point_x[1] = next_point_x[1] - 1

                    available_flags = [tuple(next_point_y) in possible_positions, tuple(next_point_x) in
                                       possible_positions]

                    if not any(available_flags):
                        return random.randint(0, 3)
                    elif all(available_flags):
                        if abs(y_del) > abs(x_del):
                            return (compass_dir_y - agent_orientation) % 4
                        else:
                            return (compass_dir_x - agent_orientation) % 4
                    else:
                        if available_flags[0]:
                            return (compass_dir_y - agent_orientation) % 4

                    return (compass_dir_x - agent_orientation) % 4

        return -1

class GreedyProbabilisticAgent(Agent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(GreedyProbabilisticAgent, self).__init__(agent_id, obs_type)
        self.color = (255, 51, 153)

    def act(self, obs):
        agent_pos = obs[0][self.agent_id]
        agent_orientation = obs[1][self.agent_id]
        oppo_pos = obs[2]
        oppo_alive_stats = obs[4]
        poss_locs = obs[5]

        pounce_res = self.pounce_prey(agent_pos, agent_orientation, oppo_pos, oppo_alive_stats)
        if pounce_res != -1:
            return pounce_res
        approach_res = self.approach_prey(agent_pos, agent_orientation, oppo_pos, oppo_alive_stats, poss_locs)
        if approach_res != -1:
            return approach_res
        else:
            return random.randint(0, 4)

    def pounce_prey(self, agent_pos, agent_orientation, oppo_pos, oppo_alive_stats):
        next_to_oppo = [self.computeManhattanDistance(agent_pos, oppo_next) < 2 for
                        idx, oppo_next in enumerate(oppo_pos) if oppo_alive_stats[idx]]
        next_to_oppo_idx = [idx for
                            idx, oppo_next in enumerate(oppo_pos) if oppo_alive_stats[idx]]

        if not any(next_to_oppo):
            return -1
        else:
            point1 = agent_pos
            point2 = oppo_pos[next_to_oppo_idx[next_to_oppo.index(True)]]
            y_del = point1[0] - point2[0]
            x_del = point1[1] - point2[1]

            compass_dir = 0
            if x_del > 0:
                compass_dir = 3
            elif x_del < 0:
                compass_dir = 1
            elif y_del < 0:
                compass_dir = 2

            real_dir = (compass_dir - agent_orientation) % 4
            return real_dir

    def approach_prey(self, agent_pos, agent_orientation, oppo_pos, oppo_alive_stats, possible_positions):
        adders = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        boundary_points = [(pos[0] + adder[0], pos[1] + adder[1])
                           for (idx, pos) in enumerate(oppo_pos)
                           for adder in adders if oppo_alive_stats[idx]
                           if (pos[0] + adder[0], pos[1] + adder[1]) in possible_positions]

        if len(boundary_points) != 0:
            manhattan_distances = [self.computeManhattanDistance(agent_pos, boundary_point)
                                   for boundary_point in boundary_points]
            min_idx = manhattan_distances.index(min(manhattan_distances))
            point1 = agent_pos
            point2 = boundary_points[min_idx]

            y_del = point1[0] - point2[0]
            x_del = point1[1] - point2[1]
            subtractor = max(abs(y_del), abs(x_del))
            y_prob = math.exp((abs(y_del) - subtractor) / 2.5) / (
                    math.exp((abs(y_del) - subtractor) / 2.5) + math.exp((abs(x_del) - subtractor) / 2.5))

            dim = 0
            if random.random() > y_prob:
                dim = 1

            v_opt = abs(y_del) + abs(x_del) - 1
            v_less_opt = abs(y_del) + abs(x_del) + 1

            if dim == 0:
                if y_del > 0:
                    opt_dest = (point1[0] - 1, point1[1])
                    sub_opt_dest = (point1[0] + 1, point1[1])
                    compass_opt = 0
                else:
                    opt_dest = (point1[0] + 1, point1[1])
                    sub_opt_dest = (point1[0] - 1, point1[1])
                    compass_opt = 2
            else:
                if x_del > 0:
                    opt_dest = (point1[0], point1[1] - 1)
                    sub_opt_dest = (point1[0], point1[1] + 1)
                    compass_opt = 3
                else:
                    opt_dest = (point1[0], point1[1] + 1)
                    sub_opt_dest = (point1[0], point1[1] - 1)
                    compass_opt = 1

            if not opt_dest in possible_positions:
                v_opt += 5

            if not sub_opt_dest in possible_positions:
                v_less_opt += 5

            subtractor = max(v_opt, v_less_opt)
            opt_prob = (math.exp((abs(v_opt) - subtractor) / -2.5)) / (
                    math.exp((abs(v_opt) - subtractor) / -2.5) + (math.exp((abs(v_less_opt) - subtractor) / -2.5)))

            dir = compass_opt
            if random.random() > opt_prob:
                dir = (compass_opt + 2) % 4

            real_dir = (dir - agent_orientation) % 4
            return real_dir

        return -1

    def computeManhattanDistance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

class GreedyProbabilisticWaitingAgent(GreedyProbabilisticAgent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(GreedyProbabilisticWaitingAgent, self).__init__(agent_id, obs_type)
        self.waiting_radius = np.random.choice(3, 1, p=[0.4, 0.4, 0.2]) + 2
        self.group_num_threshold = np.random.choice(2, 1, p=[0.7, 0.3]) + 1

    def act(self, obs):
        agent_pos = obs[0][self.agent_id]
        agent_orientation = obs[1][self.agent_id]
        oppo_pos = obs[2]
        oppo_alive_stats = obs[4]
        poss_locs = obs[5]

        in_proximity_alone = self.in_proximity_alone(agent_pos,
                                                     obs[0],
                                                     agent_orientation,
                                                     oppo_pos,
                                                     oppo_alive_stats,
                                                     poss_locs)

        if in_proximity_alone != -1:
            return in_proximity_alone
        else:
            return super(GreedyProbabilisticWaitingAgent, self).act(obs)

    def in_proximity_alone(self, agent_pos, team_pos, agent_orientation, oppo_pos, oppo_alive_stats,
                           possible_positions):

        adders = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        boundary_points = [(idx, (pos[0] + adder[0], pos[1] + adder[1]))
                           for (idx, pos) in enumerate(oppo_pos)
                           for adder in adders if oppo_alive_stats[idx]
                           if (pos[0] + adder[0], pos[1] + adder[1])
                           in possible_positions]
        manhattan_distances = [super(GreedyProbabilisticWaitingAgent, self).computeManhattanDistance(agent_pos,
                                                                                                     boundary_point[1])
                               for boundary_point in boundary_points]

        if len(manhattan_distances) != 0:
            min_idx = manhattan_distances.index(min(manhattan_distances))
            closest_agent_loc = oppo_pos[boundary_points[min_idx][0]]

            if self.computeManhattanDistance(agent_pos,
                closest_agent_loc) <= self.waiting_radius:
                waiting_teammates = 0
                for loc in team_pos:
                    if loc != agent_pos and self.computeManhattanDistance(loc,
                    closest_agent_loc) <= self.waiting_radius:
                        waiting_teammates += 1

                if waiting_teammates < self.group_num_threshold:
                    point1 = agent_pos
                    point2 = closest_agent_loc

                    y_del = point1[0] - point2[0]
                    x_del = point1[1] - point2[1]

                    next_point_x = [point1[0], point1[1]]
                    next_point_y = [point1[0], point1[1]]

                    compass_dir_y = 2
                    compass_dir_x = 3

                    if y_del < 0:
                        compass_dir_y = 0

                    if x_del > 0:
                        compass_dir_x = 1

                    if compass_dir_y == 0:
                        next_point_y[0] = next_point_y[0] - 1
                    else:
                        next_point_y[0] = next_point_y[0] + 1

                    if compass_dir_x == 1:
                        next_point_x[1] = next_point_x[1] + 1
                    else:
                        next_point_x[1] = next_point_x[1] - 1

                    available_flags = [tuple(next_point_y) in possible_positions, tuple(next_point_x) in
                                       possible_positions]

                    if not any(available_flags):
                        return random.randint(0, 3)
                    elif all(available_flags):
                        if abs(y_del) > abs(x_del):
                            return (compass_dir_y - agent_orientation) % 4
                        else:
                            return (compass_dir_x - agent_orientation) % 4
                    else:
                        if available_flags[0]:
                            return (compass_dir_y - agent_orientation) % 4

                    return (compass_dir_x - agent_orientation) % 4

        return -1

class TeammateAwarePredator(Agent):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(TeammateAwarePredator, self).__init__(agent_id, obs_type)
        self.action_distrib = None
        self.color = (0, 255, 0)

    def act(self, obs):
        agent_pos = obs[0][self.agent_id]
        all_agent_pos = obs[0]
        agent_orientation = obs[1][self.agent_id]
        oppo_pos = obs[2]
        oppo_alive_stats = obs[4]
        poss_locs = obs[5]

        adders = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        border_points = [[((point[0] + adder[0], point[1] + adder[1]), point) for adder in adders
                          if (point[0] + adder[0], point[1] + adder[1]) in poss_locs] for point in oppo_pos]
        collapsed_border = [point for elem in border_points for point in elem]
        manhattan_dists = [list(zip(collapsed_border,
                                    [self.computeManhattanDistance(agent, elem_border[0]) for elem_border in
                                     collapsed_border]))
                           for agent in all_agent_pos]

        max_manhattan_dists = [max([elem[1] for elem in k]) for k in manhattan_dists]
        enumerated_man_dist = list(enumerate(max_manhattan_dists))
        enumerated_man_dist.sort(key=(lambda x: x[1]), reverse=True)

        for a in manhattan_dists:
            a.sort(key=(lambda x: x[1]))

        dests = [None] * len(enumerated_man_dist)
        oppo_chased = [None] * len(enumerated_man_dist)
        for b in enumerated_man_dist:
            a = manhattan_dists[b[0]]
            idx = 0
            while idx < len(a) and a[idx][0][0] in dests:
                idx += 1

            if idx < len(a):
                dests[b[0]] = a[idx][0][0]
                oppo_chased[b[0]] = a[idx][0][1]

        single_dest = dests[self.agent_id]
        single_oppo_chased = oppo_chased[self.agent_id]

        if single_oppo_chased == None:
            self.action_distrib = [1.0 / 5.0] * 5
            return random.randint(0, 4)

        point1 = agent_pos
        point2 = single_oppo_chased
        y_del = point1[0] - point2[0]
        x_del = point1[1] - point2[1]

        if agent_pos == single_dest:
            compass_dir = -1
            if x_del == 1:
                compass_dir = 3
            elif x_del == -1:
                compass_dir = 1
            elif y_del == -1:
                compass_dir = 2
            elif y_del == 1:
                compass_dir = 0

            if compass_dir != -1:
                self.action_distrib = [0.0] * 5
                real_dir = (compass_dir - agent_orientation) % 4
                self.action_distrib[real_dir] = 1.0
                return real_dir

        start = agent_pos
        end = single_dest

        next_dest = self.a_star(start, end, poss_locs, all_agent_pos, oppo_pos)

        if next_dest == None:
            self.action_distrib = [1.0 / 5.0] * 5
            return random.randint(0, 4)
        point1 = agent_pos
        point2 = next_dest
        y_del = point1[0] - point2[0]
        x_del = point1[1] - point2[1]

        compass_dir = 0
        if x_del > 0:
            compass_dir = 3
        elif x_del < 0:
            compass_dir = 1
        elif y_del < 0:
            compass_dir = 2

        self.action_distrib = [0.0] * 5
        real_dir = (compass_dir - agent_orientation) % 4
        self.action_distrib[real_dir] = 1.0
        return real_dir

    def a_star(self, start, end, possible_states, teammate_locs, oppo_locs):
        frontier = Q.PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        adders = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        while not frontier.empty():
            current = frontier.get()
            if current == end:
                curr_end = current
                prev = came_from[curr_end]

                while prev != start:
                    curr_end = prev
                    prev = came_from[curr_end]
                return curr_end

            for next in [(current[0] + adder[0], current[1] + adder[1]) for adder in adders if
                         (current[0] + adder[0], current[1] + adder[1]) in possible_states
                         and (not ((current[0] + adder[0], current[1] + adder[1]) in teammate_locs))
                         and (not ((current[0] + adder[0], current[1] + adder[1]) in oppo_locs))]:
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.computeManhattanDistance(next, end)
                    frontier.put(next, priority)
                    came_from[next] = current

    def computeManhattanDistance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

class TeammateAwareWaitingAgent(TeammateAwarePredator):
    def __init__(self, agent_id, obs_type="comp_processed"):
        super(TeammateAwareWaitingAgent, self).__init__(agent_id, obs_type)
        self.waiting_radius = np.random.choice(3, 1, p=[0.4, 0.4, 0.2]) + 2
        self.group_num_threshold = np.random.choice(2, 1, p=[0.7, 0.3]) + 1

    def act(self, obs):
        agent_pos = obs[0][self.agent_id]
        agent_orientation = obs[1][self.agent_id]
        oppo_pos = obs[2]
        oppo_alive_stats = obs[4]
        poss_locs = obs[5]

        in_proximity_alone = self.in_proximity_alone(agent_pos,
                                                     obs[0],
                                                     agent_orientation,
                                                     oppo_pos,
                                                     oppo_alive_stats,
                                                     poss_locs)

        if in_proximity_alone != -1:
            return in_proximity_alone
        else:
            return super(TeammateAwareWaitingAgent, self).act(obs)

    def in_proximity_alone(self, agent_pos, team_pos, agent_orientation, oppo_pos, oppo_alive_stats,
                           possible_positions):

        adders = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        boundary_points = [(idx, (pos[0] + adder[0], pos[1] + adder[1]))
                           for (idx, pos) in enumerate(oppo_pos)
                           for adder in adders if oppo_alive_stats[idx]
                           if (pos[0] + adder[0], pos[1] + adder[1])
                           in possible_positions]
        manhattan_distances = [super(TeammateAwareWaitingAgent, self).computeManhattanDistance(agent_pos,
                                                                                  boundary_point[1])
                               for boundary_point in boundary_points]

        if len(manhattan_distances) != 0:
            min_idx = manhattan_distances.index(min(manhattan_distances))
            closest_agent_loc = oppo_pos[boundary_points[min_idx][0]]

            if super(TeammateAwareWaitingAgent, self).computeManhattanDistance(agent_pos, closest_agent_loc) \
                <= self.waiting_radius:
                waiting_teammates = 0
                for loc in team_pos:
                    if loc != agent_pos and super(TeammateAwareWaitingAgent,
                                                  self).computeManhattanDistance(loc,
                    closest_agent_loc) <= self.waiting_radius:
                        waiting_teammates += 1

                if waiting_teammates < self.group_num_threshold:
                    point1 = agent_pos
                    point2 = closest_agent_loc

                    y_del = point1[0] - point2[0]
                    x_del = point1[1] - point2[1]

                    next_point_x = [point1[0], point1[1]]
                    next_point_y = [point1[0], point1[1]]

                    compass_dir_y = 2
                    compass_dir_x = 3

                    if y_del < 0:
                        compass_dir_y = 0

                    if x_del > 0:
                        compass_dir_x = 1

                    if compass_dir_y == 0:
                        next_point_y[0] = next_point_y[0] - 1
                    else:
                        next_point_y[0] = next_point_y[0] + 1

                    if compass_dir_x == 1:
                        next_point_x[1] = next_point_x[1] + 1
                    else:
                        next_point_x[1] = next_point_x[1] - 1

                    available_flags = [tuple(next_point_y) in possible_positions, tuple(next_point_x) in
                                       possible_positions]

                    if not any(available_flags):
                        return random.randint(0, 3)
                    elif all(available_flags):
                        if abs(y_del) > abs(x_del):
                            return (compass_dir_y - agent_orientation) % 4
                        else:
                            return (compass_dir_x - agent_orientation) % 4
                    else:
                        if available_flags[0]:
                            return (compass_dir_y - agent_orientation) % 4

                    return (compass_dir_x - agent_orientation) % 4

        return -1

class DQNAgent(Agent):
    def __init__(self, agent_id, args=None, obs_type="partial_obs", obs_height=9, obs_width=17, mode="test"):
        super(DQNAgent, self).__init__(agent_id, obs_type)
        self.obs_type = obs_type
        self.args = args
        self.color = (255, 0, 0)
        self.experience_replay = ReplayMemoryLite(state_h=obs_height, state_w=obs_width,
                                                  with_gpu=self.args['with_gpu'])
        self.dqn_net = DQN(17, 9, 32, self.args['max_seq_length'], 7, mode="partial")

        if self.args['with_gpu']:
            self.dqn_net.cuda()
            self.dqn_net.device = "cuda:0"
            self.target_dqn_net.cuda()
            self.target_dqn_net.device = "cuda:0"

        self.mode = mode
        if not self.mode == "test":
            self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.args['lr'])
            self.target_dqn_net = DQN(17, 9, 32, self.args['max_seq_length'], 7, mode="partial")
            hard_copy(self.target_dqn_net, self.dqn_net)

        self.recent_obs_storage = np.zeros([self.args['max_seq_length'], obs_height, obs_width, 3])

    def load_parameters(self, filename):
        self.dqn_net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        self.dqn_net.eval()

    def save_parameters(self, filename):
        torch.save(self.dqn_net.state_dict(), filename)

    def act(self, obs, added_features=None, mode="train", epsilon=0.01):
        self.recent_obs_storage = np.roll(self.recent_obs_storage, axis=0, shift=-1)
        self.recent_obs_storage[-1] = obs
        net_inp = torch.Tensor([self.recent_obs_storage.transpose([0, 3, 1, 2])])
        _, indices = torch.max(self.dqn_net(net_inp), dim=-1)
        # Implement resets
        if not self.mode == "test":
            if random.random() < epsilon:
                indices = random.randint(0, 6)
        return indices

    def store_exp(self, exp):
        self.experience_replay.insert(exp)

    def get_obs_type(self):
        return self.obs_type

    def update(self):
        if self.experience_replay.size < self.args['sampling_wait_time']:
            return
        batched_data = self.experience_replay.sample(self.args['batch_size'])
        state, action, reward, dones, next_states = batched_data[0], batched_data[1], batched_data[2], \
                                                    batched_data[3], batched_data[4]

        state = state.permute(0, 1, 4, 2, 3)
        next_states = next_states.permute(0, 1, 4, 2, 3)

        predicted_value = self.dqn_net(state).gather(1, action.long())
        target_values = reward + self.args['disc_rate'] * (1 - dones) * torch.max(self.target_dqn_net(next_states),
                                                                                  dim=-1, keepdim=True)[0]
        loss = 0.5 * torch.mean((predicted_value - target_values.detach()) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_copy(self.target_dqn_net, self.dqn_net)


# class DistilledCoopAStarAgent(Agent):
#     def __init__(self, id, obs_type="full_graph",
#                  load_directory="dist_astar_params/DistilledNet1.pkl"):
#         super(DistilledCoopAStarAgent, self).__init__(id, obs_type)
#         self.color = (255, 128, 0)
#         self.dqn_net = GraphOppoModel(6, 0, 12, 60, 40, 50, 30, 35, 5)
#         dirname = os.path.dirname(__file__)
#         dirname = os.path.join(dirname, (load_directory))
#         self.load_params(dirname)
#         self.action_dist = None
#
#     def load_params(self, dir):
#         self.dqn_net.load_state_dict(torch.load(dir))
#
#     def act(self, obs):
#         node_obs = torch.Tensor(obs[0])
#         u_obs = torch.Tensor(obs[1])[None, :]
#
#         graph = dgl.DGLGraph()
#         num_nodes = len(obs[0])
#         graph.add_nodes(num_nodes)
#         src, dst = tuple(zip(*[(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]))
#         graph.add_edges(src, dst)
#         fin_graph = dgl.batch([graph])
#         edge_feats = torch.zeros([graph.number_of_edges(), 0])
#
#         logits_all = self.dqn_net(fin_graph, edge_feats, node_obs, u_obs)
#         m = dist.Categorical(logits=logits_all[0])
#         self.action_dist = m.probs
#
#         act = m.sample().item()
#
#         return act
#
# class GraphDQNAgent(Agent):
#     def __init__(self, id, obs_type="centralized_dqn",
#                  load_directory="dqn_gnn_params/params_40",
#                  args={'with_gpu':False}, added_u_dim=12):
#         super(GraphDQNAgent, self).__init__(id, obs_type)
#         self.args = args
#         self.color = (128,128,128)
#         self.dim_lstm_out = 50
#         self.added_u_dim = added_u_dim
#         self.device = 'cuda' if torch.cuda.is_available() and self.args['with_gpu'] else 'cpu'
#
#         self.dqn_net = RFMLSTMMiddle(2, 0, 0, 60, 40, 50, 35, 30, 40, 5, with_added_u_feat=True,
#                                     added_u_feat_dim=self.added_u_dim).to(self.device)
#         dirname = os.path.dirname(__file__)
#         dirname = os.path.join(dirname, (load_directory))
#         self.load_parameters(dirname)
#
#         self.hiddens = None
#         self.graph = None
#
#     def act(self, obs):
#         p_graph, e_ob, n_ob, u_ob, e_hiddens, n_hiddens, u_hiddens = self.prep(obs, self.hiddens)
#         batch_graph = p_graph
#
#         out, e_hid, n_hid, u_hid = self.dqn_net(batch_graph, e_ob, n_ob, u_ob,
#                                                     e_hiddens, n_hiddens, u_hiddens)
#
#         self.hiddens = (e_hid, n_hid, u_hid)
#
#         _, act = torch.max(out, dim=-1)
#         return act[self.agent_id]
#
#     def prep(self, obs, hiddens):
#         if self.graph is None:
#             self.graph = []
#             for ob in obs:
#                 graph_ob = dgl.DGLGraph()
#                 graph_ob.add_nodes(len(ob))
#                 src, dst = zip(*[(a, b) for a in range(len(ob)) for b in range(len(ob)) if a != b])
#                 graph_ob.add_edges(src, dst)
#                 self.graph.append(graph_ob)
#
#             self.graph = dgl.batch(self.graph, hiddens)
#
#         n_ob = torch.cat([torch.Tensor(elem[:2])[None, :].float() for a in obs for elem in a], dim=0)
#         e_ob = torch.zeros([self.graph.number_of_edges(), 0])
#         u_ob = torch.cat([torch.Tensor(k[0][2:]).float()[None, :] for k in obs], dim=0)
#
#         if hiddens is None:
#             n_hid = (torch.zeros([1, self.graph.number_of_nodes(), self.dim_lstm_out]),
#                     torch.zeros([1, self.graph.number_of_nodes(), self.dim_lstm_out]))
#             e_hid = (torch.zeros([1, self.graph.number_of_edges(), self.dim_lstm_out]),
#                     torch.zeros([1, self.graph.number_of_edges(), self.dim_lstm_out]))
#             u_hid = (torch.zeros([1, len(obs), self.dim_lstm_out]),
#                     torch.zeros([1, len(obs), self.dim_lstm_out]))
#
#         else:
#             n_hid = hiddens[1]
#             e_hid = hiddens[0]
#             u_hid = hiddens[2]
#
#         return self.graph, e_ob, n_ob, u_ob, e_hid, n_hid, u_hid
#
#     def reset(self):
#         self.hiddens = None
#         self.graph = None
#
#     def load_parameters(self, filename):
#         self.dqn_net.load_state_dict(torch.load(filename))