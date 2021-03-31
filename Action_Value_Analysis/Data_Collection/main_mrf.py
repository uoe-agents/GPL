from Agent import MRFAgent
from torch.utils.tensorboard import SummaryWriter
from gym.vector import AsyncVectorEnv
from datetime import date
import random
import string
import os
import json
import copy

from utils import make_open_env
from arguments import get_args

if __name__ == '__main__':

    args = vars(get_args())

    def make_env(args, rank, num_agents=5, active_agents=3, freeze_multiplier=80, team_mode="guard",
                 reward_scheme="sparse", seed=100):
        def _init():
            return make_open_env(
                args, args['num_env_steps'], num_agents, active_agents, freeze_multiplier,
                team_mode=team_mode, reward_scheme=reward_scheme, seed=int(seed + 1000 * rank)
            )

        return _init


    num_players_train = args['num_players_train']
    num_players_test = args['num_players_test']

    env = AsyncVectorEnv([
        make_env(args, i, active_agents=num_players_train, seed=args['seed'], reward_scheme=args["reward_type"]) for i
        in range(args['num_envs'])
    ])

    env_eval = AsyncVectorEnv([
        make_env(args, i, active_agents=num_players_train, seed=args['eval_init_seed'],
                 reward_scheme=args["reward_type"]) for i in range(args['num_envs'])
    ])

    env_eval2 = AsyncVectorEnv([
        make_env(args, i, active_agents=num_players_test, seed=args['eval_init_seed'],
                 reward_scheme=args["reward_type"]) for i in range(args['num_envs'])
    ])

    today = date.today()
    d1 = today.strftime("%d_%m_%Y")


    def randomString(stringLength=10):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))


    random_experiment_name = randomString(10)
    writer = SummaryWriter(log_dir="runs/" + random_experiment_name)
    directory = os.path.join(args['save_dir'], random_experiment_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    args["device"] = "cpu"
    with open(os.path.join(directory, 'params.json'), 'w') as json_file:
        json.dump(args, json_file)

    agent = MRFAgent(args=args, writer=writer, added_u_dim = 0)
    save_dirs = os.path.join(directory, 'params_0')
    agent.save_parameters(save_dirs)

    # Test at train dataset
    avgs = []
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

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
    print("Finished train with rewards " + str(avg_total_rewards))
    writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs), 0)

    avgs = []
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval2.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

    obs = env_eval2.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True)
        n_obs, rewards, dones, info = env_eval2.step(acts)
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    print("Finished test with rewards " + str(avg_total_rewards))
    writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs), 0)

    num_episode = args["max_num_steps"] // args["eps_length"]
    for ep_num in range(num_episode):
        print(ep_num)
        avgs = []
        num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']

        obs = env.reset()
        agent.reset()
        agent.set_epsilon(max(1.0 - ((ep_num + 0.0) / 2500) * 0.95, 0.05))
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

        if (ep_num + 1) % args['saving_frequency'] == 0:
            save_dirs = os.path.join(directory, 'params_' + str((ep_num +
                                                                 1) // args['saving_frequency']))
            agent.save_parameters(save_dirs)

            avgs = []
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

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
            writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

            avgs = []
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval2.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

            obs = env_eval2.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True)
                n_obs, rewards, dones, info = env_eval2.step(acts)
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
            writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])