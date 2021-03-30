import argparse
import gym
import random
import lbforaging

if __name__ == '__main__':

    def make_env(env_id, rank,  seed=1285, effective_max_num_players=3, with_shuffle=True, multi_agent=True):
        def _init():
            env = gym.make(
                env_id, seed=seed + rank,
                effective_max_num_players=effective_max_num_players,
                init_num_players=effective_max_num_players,
                with_shuffle=with_shuffle,
                multi_agent=multi_agent
            )
            return env

        return _init

    num_players_train = 3
    env = make_env('Foraging-8x8-3f-v0', 3,
        889, num_players_train, True)()

    obs = env.reset()
    actions = []

    for k in range(10000):
        actions = []
        for ob in obs :
            if ob[-1] == -1 :
                actions.append(6)
            else :
                actions.append(random.randint(0,5))
        a, b, c, d = env.step(actions)
        print(actions, a, b, c, d)
        if c:
            break
        if c:
            obs = env.reset()