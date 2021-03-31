import gym
import gym_fortattack
from gym import spaces
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from malib.spaces import Box, MASpace, MAEnvSpec
from learner import Learner
from mpnn import MPNN
from rlagent import Neo
import torch
from numpy.random import RandomState
from argparse import Namespace


def make_open_fortattack_env(args, num_steps, team_size, active_agents, num_freeze_steps, benchmark=False, seed=100,
                             reward_scheme="normal", team_mode=None, agent_type=-1):
    # Add setup methods to set number of agents for fortattack-v1
    scenario =  gym.make('fortattack-v2',
                        num_guards=team_size, num_attackers=team_size,
                        active_agents=active_agents, num_freeze_steps=num_freeze_steps,
                        seed = seed, reward_mode=reward_scheme
                )
    # create world
    world = scenario.world
    world.max_time_steps = num_steps
    # create multiagent environment
    if benchmark:
        env = OpenFortAttackGlobalEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation,
            scenario.benchmark_data, seed_callback=scenario.seed,
            arguments=args, seed=seed, with_oppo_modelling=True,
            team_mode=team_mode, agent_type=agent_type
        )
    else:
        env = OpenFortAttackGlobalEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation, seed_callback=scenario.seed,
            arguments=args, seed=seed, with_oppo_modelling=True, team_mode=team_mode, agent_type=agent_type
        )
    return env

class OpenFortAttackGlobalEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def terminate(self):
        pass

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, seed_callback=None, shared_viewer=True, arguments = {},
                 seed=100, with_oppo_modelling=False, team_mode=None, agent_type=-1):
        self.with_oppo_modelling = with_oppo_modelling
        self.ob_rms = None
        self.world = world
        self.seed_val = seed
        self.randomizer = RandomState(self.seed_val)
        self.agent_type = agent_type
        self.team_mode = team_mode

        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        if self.with_oppo_modelling:
            self.prev_actions = None
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.seed_callback = seed_callback
        self.policy_id = None
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True  # False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.ad_hoc_agent_id = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = arguments
        self.namespace_args = Namespace(**self.args)
        self.teammate_obs = None

        # configure spaces
        self.action_spaces = []
        self.observation_spaces = []
        obs_shapes = []
        self.step_num = 0
        self.agent_num = len(self.agents)
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete((world.dim_p) * 2 + 2)  ##
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_spaces.append(act_space)
            else:
                self.action_spaces.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            obs_shapes.append((obs_dim,))
            self.observation_spaces.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # action has 8 values:
        # nothing, +forcex, -forcex, +forcey, -forcey, +rot, -rot, shoot
        self.action_space = Box(low=0., high=1., shape=((world.dim_p) * 2 + 2,))  ##
        self.observation_space = Box(low=-np.inf, high=+np.inf, shape=(self.world.numAgents, obs_shapes[0][0]+2))

        if self.with_oppo_modelling:
            self.observation_space = Box(low=-np.inf, high=+np.inf, shape=(self.world.numAgents, obs_shapes[0][0] + 3))

        # Create set of learners here
        self.master = None

        #self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.action_range = [0., 1.]
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n

        self.prevShot, self.shot = False, False  # used for rendering
        self._reset_render()

    def setup_master(self):
        policy1 = None
        policy2 = None
        team1 = []  ## List of Neo objects, one Neo object per agent, teams_list = [team1, teams2]
        team2 = []

        num_adversary = 0
        num_friendly = 0
        for i, agent in enumerate(self.world.policy_agents):
            if agent.attacker:
                num_adversary += 1
            else:
                num_friendly += 1

        action_space = self.action_spaces[i]  ##* why on earth would you put i???
        pol_obs_dim = self.observation_spaces[0].shape[0]  ##* ensure 27 is correct

        # index at which agent's position is present in its observation
        pos_index = 2  ##* don't know why it's a const and +2

        for i, agent in enumerate(self.world.policy_agents):
            obs_dim = self.observation_spaces[i].shape[0]

            if not agent.attacker:  # first we have the guards and then the attackers
                if policy1 is None:
                    policy1 = MPNN(input_size=pol_obs_dim, num_agents=num_friendly, num_opp_agents=num_adversary,
                                       num_entities=0, action_space=self.action_spaces[i],
                                       pos_index=pos_index, mask_dist=1.0, entity_mp=False,
                                       policy_layers=1).to(self.device)
                team1.append(Neo(self.namespace_args, policy1, (obs_dim,),
                                     action_space))  ## Neo adds additional features to policy such as loading model, update_rollout. Neo.actor_critic is the policy and is the same object instance within a team
            else:
                if policy2 is None:
                    policy2 = MPNN(input_size=pol_obs_dim, num_agents=num_adversary, num_opp_agents=num_friendly,
                                       num_entities=0, action_space=self.action_spaces[i],
                                       pos_index=pos_index, mask_dist=1.0, entity_mp=False).to(self.device)
                team2.append(Neo(self.namespace_args, policy2, (obs_dim,), action_space))

        master = Learner(self.namespace_args, [team1, team2], [policy1, policy2], world=self.world)

        return master

    def step(self, action_n):

        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents


        masks = torch.FloatTensor(self.teammate_obs[:, 0])  ##* check the values of masks, agent alive or dead
        for agent in self.world.all_agents:
            if agent.alive:
                agent.wasAlive = True
            else:
                agent.wasAlive = False

        with torch.no_grad():
            actions_list, attn_list = self.master.act(self.step_num, masks)
        teammate_actions = np.array(actions_list).reshape(-1)
        teammate_actions[self.ad_hoc_agent_id] = action_n

        if self.with_oppo_modelling:
            self.prev_actions = np.copy(teammate_actions).reshape((-1,1))

        for i, agent in enumerate(self.agents):
            action = teammate_actions[i]
            self._set_action(action, agent, self.action_spaces[i])  # sets the actions in the agent object

        # advance world state
        ## actions are already set in the objects, so we can simply pass step without any argument
        self.world.step()  # world is the fortattack-v0 environment, step function is in core.py file

        for idx, agent in enumerate(self.agents):  ##
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            info_n['n'].append(self._get_info(agent))

        ## implement single done reflecting game state
        done = self._get_done()
        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
        self.world.time_step += 1
        self.teammate_obs = np.array(obs_n)

        swapped_obs = np.copy(self.teammate_obs)
        swapped_obs[[0, self.ad_hoc_agent_id]] = swapped_obs[[self.ad_hoc_agent_id, 0]]

        if self.with_oppo_modelling :
            swapped_prev_actions = self.prev_actions
            for idx, agent in enumerate(self.world.all_agents):
                if not agent.wasAlive:
                    swapped_prev_actions[idx][0] = -1.0
            swapped_prev_actions[[0, self.ad_hoc_agent_id]] = swapped_prev_actions[[self.ad_hoc_agent_id, 0]]

        concat_obs_n = np.concatenate((swapped_obs, self.add_team_data(swapped_obs)), axis=-1)
        if self.with_oppo_modelling:
            concat_obs_n = np.concatenate((concat_obs_n, swapped_prev_actions), axis=-1)

        rew_tens = torch.from_numpy(np.stack(reward_n)).float()
        self.master.update_rollout(self.teammate_obs, rew_tens, masks)
        self.step_num += 1


        return concat_obs_n, reward_n[self.ad_hoc_agent_id], done, info_n['n'][self.ad_hoc_agent_id]

    def add_team_data(self, obs):
        added_obs = np.zeros([obs.shape[0], 2])

        for idx in range(self.world.numAgents):
            if idx < self.world.numGuards:
                added_obs[idx, 0] = 1
            else:
                added_obs[idx, 1] = 1

        added_obs[[0, self.ad_hoc_agent_id]] = added_obs[[self.ad_hoc_agent_id, 0]]
        return added_obs

    def seed(self, seed=None):
        self.seed_val = seed
        self.randomizer = RandomState(self.seed_val)
        self.seed_callback(self.seed_val)
        return [seed]

    def reset(self):
        # reset world
        self.seed_val +=250
        self.randomizer = RandomState(self.seed_val)
        self.reset_callback()
        for agent in self.world.all_agents:
            agent.wasAlive = False

        if self.team_mode is None:
            self.ad_hoc_agent_id = self.randomizer.choice(np.concatenate(
                self.world.active_indices_guard, self.world.active_indices_attacker)
            )
        elif self.team_mode == "guard":
            self.ad_hoc_agent_id = self.randomizer.choice(self.world.active_indices_guard)
        else:
            self.ad_hoc_agent_id = self.randomizer.choice(self.world.active_indices_attacker)
        self.step_num = 0
        # reset renderer
        self._reset_render()
        # record observations for each agent
        self.policy_id = []
        self.master = self.setup_master()
        obs_n = []
        if self.with_oppo_modelling and self.prev_actions is None:
            self.prev_actions = -np.ones([self.n, 1])
        self.agents = self.world.policy_agents

        for idx, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
        self.teammate_obs = np.array(obs_n)

        swapped_data = np.copy(self.teammate_obs)
        swapped_data[[0, self.ad_hoc_agent_id]] = swapped_data[[self.ad_hoc_agent_id, 0]]
        if self.with_oppo_modelling :
            swapped_prev_actions = np.copy(self.prev_actions)
            swapped_prev_actions[[0, self.ad_hoc_agent_id]] = swapped_prev_actions[[self.ad_hoc_agent_id, 0]]

        concat_obs_n = np.concatenate((swapped_data, self.add_team_data(swapped_data)), axis=-1)
        if self.with_oppo_modelling:
            concat_obs_n = np.concatenate((concat_obs_n, swapped_prev_actions), axis=-1)

        prefix = "./marlsave"
        policies_list = []
        possible_choices = [
            "/saved_parameters_new/ep0.pt", "/tmp_1/ep220.pt", "/tmp_1/ep650.pt",
            "/tmp_1/ep1240.pt", "/tmp_1/ep1600.pt", "/tmp_1/ep2520.pt", "/tmp_2/ep5050.pt"
        ]
        for id, agent in enumerate(self.world.all_agents):
            #controller_idx = int(self.randomizer.choice(range(50))*100)
            if not agent.attacker:
                controller_idx = possible_choices[self.agent_type]
            else:
                controller_idx = possible_choices[-1]
            if self.agent_type == -1:
                controller_idx = self.randomizer.choice(possible_choices)
            self.policy_id.append(controller_idx)
            agent_pol_dir = prefix + controller_idx
            if not agent.attacker:
                sample_id = self.randomizer.choice(range(5))
            else:
                sample_id = self.randomizer.choice(range(5, 10))
            policy = torch.load(agent_pol_dir, map_location=lambda storage, loc: storage)['models'][sample_id]
            policies_list.append(policy)

        self.master.load_models(policies_list)
        self.master.set_eval_mode()
        self.master.initialize_obs(self.teammate_obs)

        for agent in self.world.all_agents:
            agent.color = np.array([0.0, 1.0, 0.0]) if not agent.attacker else np.array([1.0, 0.0, 0.0])
        self.world.all_agents[self.ad_hoc_agent_id].color[2] = 1.0


        return concat_obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get done for the whole environment
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        # done if any attacker reached landmark, attackers win
        th = self.world.fortDim
        for attacker in self.world.alive_attackers:
            dist = np.sqrt(np.sum(np.square(attacker.state.p_pos - self.world.doorLoc)))
            if dist < th:
                self.world.gameResult[2] = 1
                return (True)

            # done if max number of time steps over, guards win
        if self.world.time_step == self.world.max_time_steps - 1:
            self.world.gameResult[1] = 1
            return (True)
        elif not self.world.all_agents[self.ad_hoc_agent_id].alive :
            self.world.gameResult[3] = 1
            return (True)

        # otherwise not done
        return False

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)  ## We'll use this now for Graph NN
                # process discrete action
                ## if action[0] == 0, then do nothing
                if action[0] == 1: agent.action.u[0] = +1.0
                if action[0] == 2: agent.action.u[0] = -1.0
                if action[0] == 3: agent.action.u[1] = +1.0
                if action[0] == 4: agent.action.u[1] = -1.0
                if action[0] == 5: agent.action.u[2] = +agent.max_rot
                if action[0] == 6: agent.action.u[2] = -agent.max_rot
                agent.action.shoot = True if action[0] == 7 else False

            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]  ## each is 0 to 1, so total is -1 to 1
                    agent.action.u[1] += action[0][3] - action[0][4]  ## same as above

                    ## simple shooting action
                    agent.action.shoot = True if action[0][
                                                     6] > 0.5 else False  # a number greater than 0.5 would mean shoot

                    ## simple rotation model
                    agent.action.u[2] = 2 * (action[0][5] - 0.5) * agent.max_rot

                else:
                    agent.action.u = action[0]
            sensitivity = 5.0  # default if no value specified for accel
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u[:2] *= sensitivity

            ## remove used actions
            action = action[1:]

        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]

        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, attn_list=None, mode='human', close=False):
        # attn_list = [[teamates_attn, opp_attn] for each team]
        self.shot = False
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from gym_fortattack import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry and text for the scene
        if self.render_geoms is None or True:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from gym_fortattack import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            self.render_texts = []
            self.render_texts_xforms = []

            # add black background for active world, border of active region/world
            xMin, xMax, yMin, yMax = self.world.wall_pos
            borPts = np.array([[xMin, yMin],
                               [xMax, yMin],
                               [xMax, yMax],
                               [xMin, yMax]])
            geom = rendering.make_polygon(borPts)
            xform = rendering.Transform()
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

            geom = rendering.make_circle(self.world.fortDim)
            geom.set_color(*[0, 1, 1], alpha=1)
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(*self.world.doorLoc)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

            # ---------------- visualize attention ---------------- #
            # select reference, show attn wrt this agent
            for i, agent in enumerate(self.world.agents):
                if agent.alive or i == self.world.numGuards - 1:
                    k = i
                    break

            if self.world.vizAttn and attn_list is not None:
                # will plot attention for dead agents as well, we would know if the agents are able disregard the dead agents based of alive flag = 0
                # print('inside attn viz')
                for i, agent in enumerate(self.world.agents):
                    if agent.alive or self.world.vizDead:  # alive agents are always visualized, dead are visualized if asked
                        if i != k:
                            # if it is in the same team
                            if i < self.world.numGuards:
                                attn = attn_list[0][0][k, i]
                            else:
                                # if opponent
                                attn = attn_list[0][1][k, i - self.world.numGuards]
                                # print('attacker')
                            geom = rendering.make_circle(agent.size * (1 + attn))
                            xform = rendering.Transform()
                            geom.add_attr(xform)
                            xform.set_translation(*agent.state.p_pos)
                            alpha = 0.9 if agent.alive else 0.3
                            geom.set_color(*[1, 1, 0], alpha=alpha)
                            self.render_geoms.append(geom)
                            self.render_geoms_xform.append(xform)

            # visualize the dead agents
            if self.world.vizDead:
                for agent in self.world.agents:
                    # print(agent.name, agent.alive)
                    if not agent.alive:
                        geom = rendering.make_circle(agent.size)
                        xform = rendering.Transform()
                        geom.add_attr(xform)
                        xform.set_translation(*agent.state.p_pos)
                        geom.set_color(*agent.color, alpha=1)
                        head = rendering.make_circle(0.5 * agent.size)
                        head.set_color(*agent.color, alpha=1)
                        headXform = rendering.Transform()
                        head.add_attr(headXform)
                        shift = 0.8 * agent.size * np.array([np.cos(agent.state.p_ang), np.sin(agent.state.p_ang)])
                        headLoc = agent.state.p_pos + shift
                        headXform.set_translation(*headLoc)
                        self.render_geoms.append(head)
                        self.render_geoms_xform.append(headXform)
                        self.render_geoms.append(geom)
                        self.render_geoms_xform.append(xform)
                        self.render_texts.append(agent.name[5:])
                        self.render_texts_xforms.append(agent.state.p_pos)

            # visualize alive agents
            for entity in self.world.active_entities:  ## won't work with obstacles
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                geom.add_attr(xform)
                xform.set_translation(*entity.state.p_pos)

                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=1)
                    head = rendering.make_circle(0.5 * entity.size)
                    head.set_color(*entity.color, alpha=1)
                    headXform = rendering.Transform()
                    head.add_attr(headXform)
                    shift = 0.8 * entity.size * np.array([np.cos(entity.state.p_ang), np.sin(entity.state.p_ang)])
                    headLoc = entity.state.p_pos + shift
                    headXform.set_translation(*headLoc)
                    self.render_geoms.append(head)
                    self.render_geoms_xform.append(headXform)
                    self.render_texts.append(entity.name[5:])
                    self.render_texts_xforms.append(entity.state.p_pos)
                    # print(entity.name)

                    if entity.action.shoot:
                        self.shot = True
                        ## render the laser shots, maybe add extra delay when there is a laser shot
                        v = self.world.get_tri_pts_arr(entity)[:2, :].transpose()
                        laser = rendering.make_polygon(v)
                        laser.set_color(*entity.color, alpha=0.3)
                        laserXform = rendering.Transform()
                        laser.add_attr(laserXform)
                        self.render_geoms.append(laser)
                        self.render_geoms_xform.append(laserXform)

                else:
                    geom.set_color(*entity.color)

                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # dot for reference agent
            if self.world.vizAttn and attn_list is not None:
                # will plot attention for dead agents as well, we would know if the agents are able disregard the dead agents based of alive flag = 0
                # print('inside attn viz')
                for i, agent in enumerate(self.world.agents):
                    if agent.alive or self.world.vizDead:  # alive agents are always visualized, dead are visualized if asked
                        # select reference agent for attention viz
                        if i == k:
                            # simply put a black dot at the center of this agent
                            geom = rendering.make_circle(0.5 * agent.size)
                            xform = rendering.Transform()
                            geom.add_attr(xform)
                            xform.set_translation(*agent.state.p_pos)
                            geom.set_color(*0.5 * agent.color, alpha=1)
                            self.render_geoms.append(geom)
                            self.render_geoms_xform.append(xform)

            # add grey strips, corners of visualization window
            corPtsArr = [np.array([[xMin, yMax],
                                   [xMax, yMax],
                                   [xMax, 1],
                                   [xMin, 1]]),
                         np.array([[xMin, -1],
                                   [xMax, -1],
                                   [xMax, yMin],
                                   [xMin, yMin]])]
            for corPts in corPtsArr:
                geom = rendering.make_polygon(corPts)
                geom.set_color(*[0.5, 0.5, 0.5], alpha=1)
                xform = rendering.Transform()
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

                # add geoms to viewer ## viewer is object of class Viewer defined in rendering.py file inside gym_fortattack
            for viewer in self.viewers:
                viewer.geoms = []
                viewer.texts = []
                viewer.text_poses = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for text, xform in zip(self.render_texts, self.render_texts_xforms):
                    viewer.add_text(text, xform)

        results = []
        for i in range(len(self.viewers)):
            from gym_fortattack import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            self.viewers[i].render(return_rgb_array=False)

        self.prevShot = self.shot
        return results  ## this thing is really doing nothing

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


