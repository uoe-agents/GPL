from gym.envs.registration import register

register(
    id='wolfpack-v0',
    entry_point='Wolfpack_gym.envs:Wolfpack',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':5, 'seed':100}
)

register(
    id='wolfpack-v1',
    entry_point='Wolfpack_gym.envs:WolfpackSingle',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100, 'with_oppo_mod':True}
)

register(
    id='wolfpack-v2',
    entry_point='Wolfpack_gym.envs:WolfpackPenalty',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100}
)

register(
    id='wolfpack-v3',
    entry_point='Wolfpack_gym.envs:WolfpackPenaltySingle',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100, 'with_oppo_mod':True, 'close_penalty':0.5}
)

register(
    id='wolfpack-v4',
    entry_point='Wolfpack_gym.envs:WolfpackSingleAdhoc',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100, 'with_oppo_mod':True}
)

register(
    id='Adhoc-wolfpack-v5',
    entry_point='Wolfpack_gym.envs:WolfpackPenaltySingleAdhoc',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100, 'with_oppo_mod':True,
            'close_penalty':0.5, 'implicit_max_player_num':3, 'max_player_num':5, 'with_shuffling':True,
            'rgb_obs': False, 'tile_obs' : False, 'tile_size':5, 'rnn_with_gnn':False, 'collapsed':False}
)

register(
    id='wolfpack-v6',
    entry_point='Wolfpack_gym.envs:WolfpackPenaltySingleTrain1',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100, 'with_oppo_mod':True, 'close_penalty':0.5}
)

register(
    id='wolfpack-v7',
    entry_point='Wolfpack_gym.envs:WolfpackPenaltySingleTest1',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100, 'with_oppo_mod':True, 'close_penalty':0.5}
)

register(
    id='wolfpack-v8',
    entry_point='Wolfpack_gym.envs:WolfpackPenaltySingleTrain2',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100, 'with_oppo_mod':True, 'close_penalty':0.5}
)

register(
    id='wolfpack-v9',
    entry_point='Wolfpack_gym.envs:WolfpackPenaltySingleTest2',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100, 'with_oppo_mod':True, 'close_penalty':0.5}
)

register(
    id='MARL-wolfpack-v5',
    entry_point='Wolfpack_gym.envs:WolfpackPenaltyOpen',
    kwargs={'grid_height': 10, 'grid_width' : 10, 'num_players':3, 'seed':100,
            'close_penalty':0.5, 'implicit_max_player_num':3, 'max_player_num':5}
)
