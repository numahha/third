from my_env.hopper import MyHopperEnv1

#from gym.envs.registration import registry, register, make, spec
from gym.envs.registration import register

register(
    id='MyHopper-v2',
    entry_point='my_env:MyHopperEnv1',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

