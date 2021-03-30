from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = range(5, 11)
# players = range(2, 5)
foods = range(1, 5)
coop = [True, False]

for s, f, c in product(sizes, foods, coop):
    register(
        id="MARL-Foraging-{0}x{0}-{1}f{2}-v0".format(s, f, "-coop" if c else ""),
        entry_point="lbforaging.foraging:MARLForagingEnv",
        kwargs={
            "players": 5,
            "max_player_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 50,
            "force_coop": c,
            "seed": 100,
            "effective_max_num_players": 3,
            "init_num_players": 3,
            "with_shuffle": True,
            "gnn_input": False,
            "with_openness": True,
            "with_gnn_shuffle": False,
            "collapsed": False,
            "multi_agent": False,
        },
    )

for s, f, c in product(sizes, foods, coop):
    register(
        id="Adhoc-Foraging-{0}x{0}-{1}f{2}-v0".format(s, f, "-coop" if c else ""),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": 5,
            "max_player_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 50,
            "force_coop": c,
            "seed": 100,
            "effective_max_num_players": 3,
            "init_num_players": 3,
            "with_shuffle": True,
            "gnn_input": False,
            "with_openness": True,
            "with_gnn_shuffle": False,
            "collapsed": False,
        },
    )
