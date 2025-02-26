import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Change in Gravity: v1*

register(
    id='CartPoleWorld-v0',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 9.8},
)

register(
    id='CartPoleWorld-v1',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 24.79},
)

register(
    id='CartPoleWorld-v2',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 3.7},
)

register(
    id='CartPoleWorld-v3',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 11.15},
)

register(
    id='CartPoleWorld-v4',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 0.62},
)

register(
    id='CartPoleWorld-v10',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 2.0},
)
register(
    id='CartPoleWorld-v11',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 5.0},
)
register(
    id='CartPoleWorld-v12',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 10.0},
)
register(
    id='CartPoleWorld-v13',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 15.0},
)
register(
    id='CartPoleWorld-v14',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 20.0},
)
register(
    id='CartPoleWorld-v15',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 25.0},
)
register(
    id='CartPoleWorld-v16',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 30.0},
)
register(
    id='CartPoleWorld-v17',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 35.0},
)
register(
    id='CartPoleWorld-v18',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 40.0},
)
register(
    id='CartPoleWorld-v19',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 45.0},
)
register(
    id='CartPoleWorld-v20',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 50.0},
)
register(
    id='CartPoleWorld-v21',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 55.0},
)

# Change in CartMass: v2*


register(
    id='CartPoleWorld-v30',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 0.2},
)
register(
    id='CartPoleWorld-v31',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 0.5},
)
register(
    id='CartPoleWorld-v32',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 1.0},
)
register(
    id='CartPoleWorld-v33',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 1.5},
)
register(
    id='CartPoleWorld-v34',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 2.0},
)
register(
    id='CartPoleWorld-v35',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 2.5},
)
register(
    id='CartPoleWorld-v36',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 3.0},
)
register(
    id='CartPoleWorld-v37',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 3.5},
)
register(
    id='CartPoleWorld-v38',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 4.0},
)
register(
    id='CartPoleWorld-v39',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 4.5},
)
register(
    id='CartPoleWorld-v40',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 5.0},
)
register(
    id='CartPoleWorld-v41',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 5.5},
)

# Change in friction (nonstationary)
register(
    id='CartPoleWorld-v50',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 9.8},
)
