import argparse
import gym
import random
import Wolfpack_gym
from Agent import MRFAgent
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from gym.vector import AsyncVectorEnv
from datetime import date
import random
import string
import os
import json
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='dicount_rate')
parser.add_argument('--num_episodes', type=int, default=2000, help="Number of episodes for training")
parser.add_argument('--update_frequency', type=int, default=4, help="Timesteps between updates")
parser.add_argument('--saving_frequency', type=int,default=50,help="saving frequency")
parser.add_argument('--with_gpu', type=bool,default=False,help="with gpu")
parser.add_argument('--num_envs', type=int,default=16, help="Number of environments")
parser.add_argument('--tau', type=float,default=0.001, help="tau")
parser.add_argument('--clip_grad', type=float,default=10.0, help="gradient clipping")
parser.add_argument('--eval_eps', type=int, default=5, help="Evaluation episodes")
parser.add_argument('--weight_predict', type=float, default=1.0, help="Evaluation episodes")
parser.add_argument('--save_dir', type=str, default='parameters', help="parameter dir name")
parser.add_argument('--num_players', type=int, default=3, help="num players")
parser.add_argument('--pair_comp', type=str, default='bmm', help="pairwise factor computation")
parser.add_argument('--info', type=str, default="", help="additional info")
parser.add_argument('--seed', type=int, default=0, help="additional info")
parser.add_argument('--close_penalty', type=float, default=0.1, help="close penalty")
args = parser.parse_args()

if __name__ == '__main__':

    args = vars(args)
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")

    def randomString(stringLength=10):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))

    random_experiment_name = randomString(10)
    writer = SummaryWriter(log_dir="runs/"+random_experiment_name)
    directory = os.path.join(args['save_dir'], random_experiment_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory,'params.json'), 'w') as json_file:
        json.dump(args, json_file)

    with open(os.path.join('runs',random_experiment_name, 'params.json'), 'w') as json_file:
        json.dump(args, json_file)

    # Initialize GPL-Q Agent
    agent = MRFAgent(args=args, writer=writer, added_u_dim = 12)

    # Define the training environment
    def make_env(env_id, rank, num_players, seed=1285, close_penalty=0.5, implicit_max_player_num=3, with_shuffling=False):
        def _init():
            env = gym.make(env_id, seed=seed + rank, num_players=num_players, close_penalty=close_penalty, implicit_max_player_num=implicit_max_player_num,  with_shuffling=with_shuffling)
            return env

        return _init


    num_players = args['num_players']
    env = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  args['seed'], args['close_penalty']) for i in range(args['num_envs'])])

    # Save initial model parameters.
    save_dirs = os.path.join(directory, 'params_0')
    agent.save_parameters(save_dirs)

    # Evaluate initial model performance in training environment
    avgs = []
    for ep_val_num in range(args['eval_eps']):
        num_players = args['num_players']
        agent.reset()
        steps = 0
        avg_total_rewards = 0.0
        env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty']) for i in range(args['num_envs'])])

        f_done = False
        obs = env_eval.reset()

        while not f_done:
            acts = agent.step(obs, eval=True)
            n_obs, rewards, dones, info = env_eval.step(acts)
            avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
            f_done = any(dones)
            obs = n_obs
        avgs.append(avg_total_rewards)
        print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),0)

    # Evaluate initial model performance in test environment
    avgs = []
    for ep_val_num in range(args['eval_eps']):
        num_players = args['num_players']
        agent.reset()
        steps = 0
        avg_total_rewards = 0.0
        env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty'], implicit_max_player_num=5) for i in range(args['num_envs'])])

        f_done = False
        obs = env_eval.reset()

        while not f_done:
            acts = agent.step(obs, eval=True)
            n_obs, rewards, dones, info = env_eval.step(acts)
            avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
            f_done = any(dones)
            obs = n_obs
        avgs.append(avg_total_rewards)
        print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),0)

    # Agent training loop
    for ep_num in range(args['num_episodes']):
        print(ep_num)

        train_avgs = 0
        steps = 0
        f_done = False

        # Reset agent hidden vectors at the beginning of each episode.
        agent.reset()
        agent.set_epsilon(max(1.0 - ((ep_num + 0.0) / 1500) * 0.95, 0.05))
        obs = env.reset()
        agent.compute_target(None, None, None, None, obs, add_storage=False)

        while not f_done:
            acts = agent.step(obs)
            n_obs, rewards, dones, info = env.step(acts)
            f_done = any(dones)

            n_obs_replaced = n_obs
            if f_done:
                n_obs_replaced = copy.deepcopy(n_obs)
                for key in n_obs_replaced.keys():
                    for idx in range(len(n_obs_replaced[key])):
                        if dones[idx]:
                            n_obs_replaced[key][idx] = info[idx]['terminal_observation'][key]
            steps += 1

            train_avgs += (sum(rewards) + 0.0) / len(rewards)
            agent.compute_target(obs, acts, rewards, dones, n_obs_replaced, add_storage=True)

            if steps % args['update_frequency'] == 0 or f_done:
                agent.update()

            obs = n_obs

        writer.add_scalar('Rewards/train', train_avgs, ep_num)

        # Checkpoint and evaluate agents every few episodes.
        if (ep_num + 1) % args['saving_frequency'] == 0:
            save_dirs = os.path.join(directory, 'params_'+str((ep_num +
                                                               1) // args['saving_frequency']))
            agent.save_parameters(save_dirs)

            # Run evaluation in training environment.
            avgs = []
            for ep_val_num in range(args['eval_eps']):
                num_players = args['num_players']
                agent.reset()
                steps = 0
                avg_total_rewards = 0.0
                env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty']) for i in range(args['num_envs'])])

                f_done = False
                obs = env_eval.reset()

                while not f_done:
                    acts = agent.step(obs, eval=True)
                    n_obs, rewards, dones, info = env_eval.step(acts)
                    avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
                    f_done = any(dones)
                    obs = n_obs
                avgs.append(avg_total_rewards)
                print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

            # Run evaluation in testing environment.
            avgs = []
            for ep_val_num in range(args['eval_eps']):
                num_players = args['num_players']
                agent.reset()
                steps = 0
                avg_total_rewards = 0.0
                env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty'], implicit_max_player_num=5) for i in range(args['num_envs'])])

                f_done = False
                obs = env_eval.reset()

                while not f_done:
                    acts = agent.step(obs, eval=True)
                    n_obs, rewards, dones, info = env_eval.step(acts)
                    avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
                    f_done = any(dones)
                    obs = n_obs
                avgs.append(avg_total_rewards)
                print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

