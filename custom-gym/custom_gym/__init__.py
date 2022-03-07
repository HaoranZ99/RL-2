from gym.envs.registration import (
    registry,
    register,
)

register(
    id="Island-v0",
    entry_point="custom_gym.envs:IslandEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)