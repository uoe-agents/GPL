from Agent import MRFAgent
from torch.utils.tensorboard import SummaryWriter
from gym.vector import AsyncVectorEnv
from datetime import date
import random
import string
import os
import json
import copy
import math

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

    # Initialize training & evaluation environments.
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

    # Create logging and checkpointing folders.
    random_experiment_name = randomString(10)
    writer = SummaryWriter(log_dir="runs/" + random_experiment_name)
    directory = os.path.join(args['save_dir'], random_experiment_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    args["device"] = "cpu"
    with open(os.path.join(directory, 'params.json'), 'w') as json_file:
        json.dump(args, json_file)

    agent = MRFAgent(args=args, writer=writer, added_u_dim = 0, temp=args["init_temp"])
    save_dirs = os.path.join(directory, 'params_0')
    agent.save_parameters(save_dirs)

    # Evaluate initial parameter at training environment
    avgs = []
    num_hits, num_shoots = 0, 0
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

    obs = env_eval.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True)
        num_shoots += sum([act == 7 for act in acts])
        n_obs, rewards, dones, info = env_eval.step(acts)
        num_hits += sum([rew > 1.8 and rew < 9.2 for rew in rewards])
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    shoot_accuracy = (num_hits + 0.0) / num_shoots if num_shoots != 0 else 0.0
    print("Finished train with rewards " + str(avg_total_rewards))
    writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs), 0)
    writer.add_scalar('Shooting/train_accuracy', shoot_accuracy, 0)
    writer.add_scalar('Shooting/train_hits', num_hits, 0)
    writer.add_scalar('Shooting/train_attempts', num_shoots, 0)

    avgs = []
    num_hits, num_shoots = 0, 0
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

    obs = env_eval.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True, hard_eval=True)
        num_shoots += sum([act == 7 for act in acts])
        n_obs, rewards, dones, info = env_eval.step(acts)
        num_hits += sum([rew > 1.8 and rew < 9.2 for rew in rewards])
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    shoot_accuracy = (num_hits + 0.0) / num_shoots if num_shoots != 0 else 0.0
    print("Finished train with rewards " + str(avg_total_rewards))
    writer.add_scalar('Rewards/train_set_hard', sum(avgs) / len(avgs), 0)
    writer.add_scalar('Shooting/train_accuracy_hard', shoot_accuracy, 0)
    writer.add_scalar('Shooting/train_hits_hard', num_hits, 0)
    writer.add_scalar('Shooting/train_attempts_hard', num_shoots, 0)

    # Evaluate initial parameter at testing environment
    avgs = []
    num_hits, num_shoots = 0, 0
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval2.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

    obs = env_eval2.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True)
        num_shoots += sum([act == 7 for act in acts])
        n_obs, rewards, dones, info = env_eval2.step(acts)
        num_hits += sum([rew > 1.8 and rew < 9.2 for rew in rewards])
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    shoot_accuracy = (num_hits + 0.0) / num_shoots if num_shoots != 0 else 0.0
    print("Finished test with rewards " + str(avg_total_rewards))
    writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs), 0)
    writer.add_scalar('Shooting/eval_accuracy', shoot_accuracy, 0)
    writer.add_scalar('Shooting/eval_hits', num_hits, 0)
    writer.add_scalar('Shooting/eval_attempts', num_shoots, 0)

    avgs = []
    num_hits, num_shoots = 0, 0
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    env_eval2.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

    obs = env_eval2.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True, hard_eval=True)
        num_shoots += sum([act == 7 for act in acts])
        n_obs, rewards, dones, info = env_eval2.step(acts)
        num_hits += sum([rew > 1.8 and rew < 9.2 for rew in rewards])
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    shoot_accuracy = (num_hits + 0.0) / num_shoots if num_shoots != 0 else 0.0
    print("Finished test with rewards " + str(avg_total_rewards))
    writer.add_scalar('Rewards/eval_hard', sum(avgs) / len(avgs), 0)
    writer.add_scalar('Shooting/eval_accuracy_hard', shoot_accuracy, 0)
    writer.add_scalar('Shooting/eval_hits_hard', num_hits, 0)
    writer.add_scalar('Shooting/eval_attempts_hard', num_shoots, 0)

    # Training loop
    num_episode = args["max_num_steps"] // args["eps_length"]
    for ep_num in range(num_episode):
        print(ep_num)
        avgs = []

        num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']

        obs = env.reset()
        agent.reset()
        if not args['constant_temp']:
            # Anneal temperature for Boltzmann distribution
            if not args['temp_annealing'] == "linear":
                start_log = math.log(args['init_temp'])
                end_log = math.log(args['final_temp'])
                agent.set_temp(math.exp(max(start_log - ((ep_num + 0.0) / 2500) * (start_log - end_log), end_log)))
            else:
                agent.set_temp(
                    max(args['init_temp'] - ((ep_num + 0.0) / 2500) * (args['init_temp'] - args['final_temp']),
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

        # Checkpoint model parameters
        if (ep_num + 1) % args['saving_frequency'] == 0:
            save_dirs = os.path.join(directory, 'params_' + str((ep_num +
                                                                 1) // args['saving_frequency']))
            agent.save_parameters(save_dirs)

            # Evaluate checkpointed model parameters in training environment.
            avgs = []
            num_hits, num_shoots = 0, 0
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

            obs = env_eval.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True)
                num_shoots += sum([act == 7 for act in acts])
                n_obs, rewards, dones, info = env_eval.step(acts)
                num_hits += sum([rew > 1.8 and rew < 9.2 for rew in rewards])
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            shoot_accuracy = (num_hits + 0.0) / num_shoots if num_shoots != 0 else 0.0
            print("Finished eval with rewards " + str(avg_total_rewards))
            writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/train_accuracy', shoot_accuracy, (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/train_hits', num_hits, (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/train_attempts', num_shoots, (ep_num + 1) // args['saving_frequency'])

            avgs = []
            num_hits, num_shoots = 0, 0
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

            obs = env_eval.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True, hard_eval=True)
                num_shoots += sum([act == 7 for act in acts])
                n_obs, rewards, dones, info = env_eval.step(acts)
                num_hits += sum([rew > 1.8 and rew < 9.2 for rew in rewards])
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            shoot_accuracy = (num_hits + 0.0) / num_shoots if num_shoots != 0 else 0.0
            print("Finished eval with rewards " + str(avg_total_rewards))
            writer.add_scalar('Rewards/train_set_hard', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/train_accuracy_hard', shoot_accuracy, (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/train_hits_hard', num_hits, (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/train_attempts_hard', num_shoots, (ep_num + 1) // args['saving_frequency'])

            # Evaluate checkpointed model parameters in testing environment.
            avgs = []
            num_hits, num_shoots = 0, 0
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval2.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

            obs = env_eval2.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True)
                num_shoots += sum([act == 7 for act in acts])
                n_obs, rewards, dones, info = env_eval2.step(acts)
                num_hits += sum([rew > 1.8 and rew < 9.2 for rew in rewards])
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            shoot_accuracy = (num_hits + 0.0) / num_shoots if num_shoots != 0 else 0.0
            print("Finished eval with rewards " + str(avg_total_rewards))
            writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/eval_accuracy', shoot_accuracy, (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/eval_hits', num_hits, (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/eval_attempts', num_shoots, (ep_num + 1) // args['saving_frequency'])

            avgs = []
            num_hits, num_shoots = 0, 0
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            env_eval2.seed([args['eval_init_seed'] + 1000 * rank for rank in range(args['num_envs'])])

            obs = env_eval2.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True, hard_eval=True)
                num_shoots += sum([act == 7 for act in acts])
                n_obs, rewards, dones, info = env_eval2.step(acts)
                num_hits += sum([rew > 1.8 and rew < 9.2 for rew in rewards])
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            shoot_accuracy = (num_hits + 0.0) / num_shoots if num_shoots != 0 else 0.0
            print("Finished eval with rewards " + str(avg_total_rewards))
            writer.add_scalar('Rewards/eval_hard', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/eval_accuracy_hard', shoot_accuracy, (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/eval_hits_hard', num_hits, (ep_num + 1) // args['saving_frequency'])
            writer.add_scalar('Shooting/eval_attempts_hard', num_shoots, (ep_num + 1) // args['saving_frequency'])