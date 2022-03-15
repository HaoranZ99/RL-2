# ref: https://stable-baselines.readthedocs.io/en/master/guide/examples.html
import gym
import custom_gym

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

env = gym.make('Island-v0')

# Instantiate the agent
model = DQN(MlpPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=2e5)
print('Train done.')
# Save the agent
model.save("deepq_island")

del model # remove to demonstrate saving and loading

# Load the trained agent
model = DQN.load("deepq_island")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, rewards, dones, _ = env.step(action)
    print("rewards:%3.2f" % rewards)