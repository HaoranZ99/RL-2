import gym

import custom_gym

env = gym.make('Island-v0')

def test_env():
    print(env.action_space)
    print(env.observation_space)

    env.reset()
    print(env.state)

def random_moves():
    env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action=action)
    print(f"If we take action {action}, then the obs is {obs}, reward is {reward}")

def main():
    for i in range(10):
        random_moves()

if __name__ == '__main__':
    main()
