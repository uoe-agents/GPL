import argparse
import gym
import random
import lbforaging
from Agent import MRFAgent
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from gym.vector import AsyncVectorEnv
from datetime import date
import random
import string
import os
import json
import math

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount_rate.')
parser.add_argument('--max_num_steps', type=int, default=400000, help="Number of episodes for training.")
parser.add_argument('--eps_length', type=int, default=200, help="Episode length for training")
parser.add_argument('--update_frequency', type=int, default=4, help="Timesteps between updates.")
parser.add_argument('--saving_frequency', type=int,default=50,help="Number of episodes between checkpoints.")
parser.add_argument('--num_envs', type=int,default=16, help="Number of parallel environments for training.")
parser.add_argument('--tau', type=float,default=0.001, help="Tau for soft target update.")
parser.add_argument('--eval_eps', type=int, default=5, help="Number of episodes for evaluation.")
parser.add_argument('--weight_predict', type=float, default=1.0, help="Weight associated to action prediction loss.")
parser.add_argument('--save_dir', type=str, default='parameters', help="Directory name for saving parameters.")
parser.add_argument('--num_players_train', type=int, default=3, help="Maximum number of players for training.")
parser.add_argument('--num_players_test', type=int, default=5, help="Maximum number of players for testing.")
parser.add_argument('--pair_comp', type=str, default='bmm', help="Pairwise factor computation method. Use bmm for low rank factorization.")
parser.add_argument('--info', type=str, default="", help="Additional info.")
parser.add_argument('--seed', type=int, default=0, help="Training seed.")
parser.add_argument('--eval_init_seed', type=int, default=2500, help="Evaluation seed.")
parser.add_argument('--constant_temp', type=bool, default=False, help="Constant temperature is used if set to True.")
parser.add_argument('--init_temp', type=float, default=1.0, help="Initial temperature for Boltzmann distribution.")
parser.add_argument('--final_temp', type=float, default=0.01, help="Final temperature for Boltzmann distribution.")
parser.add_argument('--temp_annealing', type=str, default="linear", help="Temperature annealing method.")

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

    # Initialize the GPL-SPI Agent
    agent = MRFAgent(args=args, writer=writer, added_u_dim = 9, temp=args["init_temp"])

    # Define the training environment
    num_players_train = args['num_players_train']
    num_players_test = args['num_players_test']

    def make_env(env_id, rank, seed=1285, effective_max_num_players=3, with_shuffle=False, gnn_input=True):
        def _init():
            env = gym.make(
                env_id, seed=seed + rank,
                effective_max_num_players=effective_max_num_players,
                init_num_players=effective_max_num_players,
                with_shuffle=with_shuffle,
                gnn_input=gnn_input
            )
            return env

        return _init


    env = AsyncVectorEnv(
        [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                  args['seed'], num_players_train, False, True)
         for i in range(args['num_envs'])]
    )

    # Save init agent model parameters.
    save_dirs = os.path.join(directory, 'params_0')
    agent.save_parameters(save_dirs)

    # Evaluate initial model performance in training environment
    avgs = []
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval = AsyncVectorEnv(
        [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                  args['eval_init_seed'], num_players_train, False, True)
         for i in range(args['num_envs'])]
    )

    obs = env_eval.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True)
        n_obs, rewards, dones, info = env_eval.step(acts)
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs
        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs), 0)

    # Evaluate initial model performance in training environment when temperature approaches 0
    avgs = []
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval = AsyncVectorEnv(
        [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                  args['eval_init_seed'], num_players_train, False, True)
         for i in range(args['num_envs'])]
    )

    obs = env_eval.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True, hard_eval=True)
        n_obs, rewards, dones, info = env_eval.step(acts)
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs
        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/train_set_hard', sum(avgs) / len(avgs), 0)

    # Evaluate initial model performance in test environment
    avgs = []
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval = AsyncVectorEnv(
        [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                  args['eval_init_seed'], num_players_test, False, True)
         for i in range(args['num_envs'])]
    )

    obs = env_eval.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True)
        n_obs, rewards, dones, info = env_eval.step(acts)
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs), 0)

    avgs = []
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval = AsyncVectorEnv(
        [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                  args['eval_init_seed'], num_players_test, False, True)
         for i in range(args['num_envs'])]
    )

    # Evaluate initial model performance in test environment when temperature approaches 0.
    obs = env_eval.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True, hard_eval=True)
        n_obs, rewards, dones, info = env_eval.step(acts)
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/eval_hard', sum(avgs) / len(avgs), 0)

    # Agent training loop
    num_episode = args["max_num_steps"] // args["eps_length"]
    for ep_num in range(num_episode):
        print(ep_num)

        # Store performance stats during training
        avgs = []
        num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']

        # Reset agent hidden vectors at the beginning of each episode.
        obs = env.reset()
        agent.reset()
        if not args['constant_temp']:
            # Anneal temperature parameters at each episode.
            if not args['temp_annealing'] == "linear":
                start_log = math.log(args['init_temp'])
                end_log = math.log(args['final_temp'])
                agent.set_temp(math.exp(max(start_log - ((ep_num + 0.0) / 1500) * (start_log - end_log), end_log)))
            else:
                agent.set_temp(
                    max(args['init_temp'] - ((ep_num + 0.0) / 1500) * (args['init_temp'] - args['final_temp']),
                        args['final_temp']))
        agent.compute_target(None, None, None, None, obs, add_storage=False)
        steps = 0

        while steps < args["eps_length"]:
            acts = agent.step(obs)
            n_obs, rewards, dones, info = env.step(acts)
            per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
            agent.compute_target(obs, acts, rewards, dones, n_obs, add_storage=True)
            obs = n_obs

            # Compute updated reward
            for idx, flag in enumerate(dones):
                if flag:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                    per_worker_rew[idx] = 0

            steps += 1
            if steps % args['update_frequency'] == 0:
                agent.update()

        train_avgs = (sum(avgs) + 0.0) / len(avgs) if len(avgs) != 0 else 0.0
        writer.add_scalar('Rewards/train', train_avgs, ep_num)

        # Checkpoint and evaluate agents every few episodes.
        if (ep_num + 1) % args['saving_frequency'] == 0:
            save_dirs = os.path.join(directory, 'params_'+str((ep_num +
                                                               1) // args['saving_frequency']))
            agent.save_parameters(save_dirs)

            # Run evaluation in training environment.
            avgs = []
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval = AsyncVectorEnv(
                [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                          args['eval_init_seed'], num_players_train, False, True)
                 for i in range(args['num_envs'])]
            )

            obs = env_eval.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True)
                n_obs, rewards, dones, info = env_eval.step(acts)
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

            # Run evaluation in training environment when temperature approaches 0.
            avgs = []
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval = AsyncVectorEnv(
                [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                          args['eval_init_seed'], num_players_train, False, True)
                 for i in range(args['num_envs'])]
            )

            obs = env_eval.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True, hard_eval=True)
                n_obs, rewards, dones, info = env_eval.step(acts)
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/train_set_hard', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

            # Run evaluation in testing environment.
            avgs = []
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval = AsyncVectorEnv(
                [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                          args['eval_init_seed'], num_players_test, False, True)
                 for i in range(args['num_envs'])]
            )

            obs = env_eval.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True)
                n_obs, rewards, dones, info = env_eval.step(acts)
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

            # Run evaluation in testing environment when temperature approaches 0.
            avgs = []
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval = AsyncVectorEnv(
                [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                          args['eval_init_seed'], num_players_test, False, True)
                 for i in range(args['num_envs'])]
            )

            obs = env_eval.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True, hard_eval=True)
                n_obs, rewards, dones, info = env_eval.step(acts)
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/eval_hard', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])
