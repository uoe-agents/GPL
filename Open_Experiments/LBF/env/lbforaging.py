import argparse
import logging
import random
import time
import gym
import numpy as np
import lbforaging.foraging
from MADummyVecEnv import MADummyVecEnv
from lbforaging.agents import H1, H2, H3, H4
import sys
np.set_printoptions(threshold=sys.maxsize)


logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """
    """
    obs = env.reset()
    done = False

    agent1 = H2(env.players[0])
    agent2 = H3(env.players[1])

    if render:
        env.render()
        time.sleep(0.5)

    while not done:

        actions = []

        obs1 = obs[0][0]
        obs2 = obs[1][0]

        actions = [agent1._step(obs1), agent2._step(obs2)]

        nobs, nreward, ndone, _ = env.step(actions)
        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            time.sleep(0.5)

        obs = nobs
        done = np.all(ndone)
    # print(env.players[0].score, env.players[1].score)


def main(game_count=1, render=False):
    env = gym.make("Foraging-10x10-2p-4f-v0")
    obs = env.reset()

    for episode in range(game_count):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )
    parser.add_argument(
        "--num_envs", type=int, default=2, help="Number of parallel envs"
    )

    args = parser.parse_args()
    # main(args.times, args.render)
    import time
    import random


    def make_env(env_id, rank, seed=1285):
        def _init():
            env = gym.make(env_id, seed=seed + rank)
            return env

        return _init

    env = MADummyVecEnv([make_env("Foraging-8x8-3f-v0", rank=idx, seed=100) for idx in range(args.num_envs)])
    print(env.observation_space)
    obs = env.reset()
    for a in range(100):
        #env.render()
        acts = [[asp.sample() for asp in env.action_space] for _ in range(args.num_envs)]
        obs, rew, done, info = env.step(acts)
        print(obs[0][:,-1])
        print(obs[0][:,-1].sum())
        exit()
