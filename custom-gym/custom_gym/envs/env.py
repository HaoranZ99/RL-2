"""
Simulation of a simple environment for the islanders to interact.
"""
import math
from time import sleep
from typing import Optional, Union

import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding
from pyparsing import replaceWith

import logging

class IslandEnv(gym.Env):
    """
    ### Description
    An agent is on a group of islands containing multiple ai's, the goal is to be able to live a little longer.

    ### Action Space
    The agent take a 1-element vector for actions.
    Actions:
    | Num | Action                        |
    |-----|-------------------------------|
    | 0   | Give gift to another islander |
    | 1   | Rab another islander          |
    | 2   | Do nothing                    |

    ### Observation Space
    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
    | Num | Observation           | Min                  | Max                |
    |-----|-----------------------|----------------------|--------------------|
    | 0   | Islander Hp           | 0                    | Inf                |
    | 1   | Islander Money        | 0                    | Inf                |


    ### Rewards
    Reward is 1 for every step taken

    ### Starting state
    Islander's hp is set to 100 and its money is set to 10

    ### Episode Termination
    The episode terminates of one of the following occurs:

    1. Island's hp is below 1.
    2. Episode length is greater than 200.   
    """

    # No metadata needed.

    def __init__(self):
        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32)

        self.state = None

    def step(self, action):
        hp, money = self.state

        reward = 1

        if action == 0:
            """
            Ai may be unhappy after receiving the gift, causing agent's life value loss.
            - First set the possibility of happy as 0.6, no feeling as 0.3, unhappy as 0.1.
            - happy life value increases by 1, unhappy life value decreases by 3.
            """
            """显式地对其奖励"""
            money -= 1
            hp += 1

        elif action == 1:
            """
            Ai being stolen money will (with probability) hit agent, causing agent's life value to be lost.
            - The probability of being hit is 0.7.
            - The life value of being hit is reduced by 8.
            - 显式地对其惩罚
            """
            money += 5
            hp -= 8

        else:
            """
            Agent does not do anything will suffer continuous life loss.
            - Not doing anything reduces life by 10.
            - 显式地对其惩罚
            """
            hp -= 10
            reward = -5
        
        if hp < 20 or money < 3:
            reward -= 2
            
        self.state = (hp, money)

        # """
        # Simple logs to log each step
        # """
        # logging.basicConfig(
        #     level=logging.INFO,
        #     filename='./logs/3-10-14_45.log',
        #     filemode='a',
        #     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        # )
        # logging.info(f"Last action chosen, {action}. Current state: hp, {hp}, money, {money}")

        done = bool(
            hp < 0
            or money < 0
        )
        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def reset(
        self
    ):
        # super().reset()
        self.state = (100.0, 10.0)
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
