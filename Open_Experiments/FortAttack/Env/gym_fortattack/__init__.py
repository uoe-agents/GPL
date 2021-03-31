from gym.envs.registration import register
 
register(id='fortattack-v0', 
    entry_point='gym_fortattack.envs:FortAttackEnv', 
)
register(id='fortattack-v1', 
    entry_point='gym_fortattack.envs:FortAttackEnvV1', 
)
register(id='fortattack-v2',
    entry_point='gym_fortattack.envs:FortAttackEnvV2',
    kwargs={'num_guards': 5, 'num_attackers' : 5, 'seed':100, 'num_freeze_steps':60, "reward_mode":"normal",
            "active_agents":5}
)