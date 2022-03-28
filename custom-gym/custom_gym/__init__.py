from gym.envs.registration import (
    registry,
    register,
)

register(
    id="Island-v0",
    entry_point="custom_gym.envs:IslandEnv",
    max_episode_steps=500,
    reward_threshold=4950.0,
)