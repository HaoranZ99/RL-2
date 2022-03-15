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
    | 0   | Islander Hp           | -Inf                 | Inf                |
    | 1   | Islander Money        | -Inf                 | Inf                |


    ### Rewards
    Reward is 1 for every step taken.
    - If agent chooses to do nothing, hp - 1.
    - If agent's hp or money is too low, it receives continuous punishment.

    ### Starting state
    Islander's hp is set to 100 and its money is set to 10

    ### Episode Termination
    The episode terminates of one of the following occurs:

    1. Islander's hp is below 0.
    2. Islander's money is below 0.
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

        if action == 0:
            money -= 1
            hp += 1
            reward = 1

        elif action == 1:
            money += 5
            hp -= 8
            reward = -1

        else:
            hp -= 10
            reward = -5
        
        if hp < 20 or money < 3:
            reward -= 2
            
        self.state = (hp, money)

        done = bool(
            hp < 0
            or money < 0
        )
        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def reset(
        self
    ):
        self.state = (100.0, 10.0)
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
