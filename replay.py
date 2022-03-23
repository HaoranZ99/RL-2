import random, datetime
from pathlib import Path

import gym
import custom_gym

from logger import Logger
from islandersurvive import DQN
env = gym.make('Island-v0')

env.reset()
save_dir = Path('replays') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/2022-03-22T19-00-12/island_net_5.chkpt')
dqn = DQN(save_dir=save_dir, checkpoint=checkpoint)
dqn.epsilon = dqn.epsilon_min

logger = Logger(save_dir)
dict = logger.get_action_meanings()

episodes = 1000

for e in range(episodes):

    s = env.reset()

    logger.replay_log(save_dir, 'replay_log', 'States initialized.')
    while True:
        a = dqn.choose_action(s)

        # 选动作, 得到环境反馈
        s_, r, done, _ = env.step(a)

        # display step
        logger.replay_log_step(save_dir, 'replay_log', s_, dict.get(a))

        # 存记忆
        dqn.store_transition(s, a, r, s_)

        logger.log_step(r, None, None)

        if done:
            logger.replay_log(save_dir, 'replay_log', f"One episode ends.")
            break

        s = s_ # 将下一个state的值传到下一次循环

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=dqn.epsilon,
            step=dqn.learn_step_counter
        )