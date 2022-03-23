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
    | 0   | Eat                           |
    | 1   | Give gift to another islander |
    | 2   | Do nothing                    |
    | 3   | Chat with another islander    |
    | 4   | Work                          |
    | 5   | Rob another islander          |

    ### Observation Space
    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
    | Num | Observation           | Min                  | Max                |
    |-----|-----------------------|----------------------|--------------------|
    | 0   | Islander Health       | -Inf                 | Inf                |
    | 1   | Islander Reputation   | -Inf                 | Inf                |
    | 2   | Islander Money        | -Inf                 | Inf                |

    ### Rewards
    Reward is 1 for every step taken except for action no.2 and no.5.
    - If agent chooses to do nothing, reward = -5.
    - If agent chooses to rob, reward = -1.
    - If agent's health or money is too low, it receives continuous punishment, (reward -= 2).

    ### Starting state
    Islander's health is set to 100 and its money is set to 10 and its reputaiton is set to 10.

    ### Episode Termination
    The episode terminates of one of the following occurs:

    1. Islander's health is below 0.
    2. Islander's reputation is below 0.
    3. Islander's money is below 0.
    2. Episode length is greater than 200.   
    """

    # No metadata needed.

    def __init__(self):
        high = np.array(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32)

        self.state = None

    def step(self, action):
        health, reputation, money = self.state
        reward = 1

        if action == 0: # Eat
            health += 10
            money -= 1
        elif action == 1: # Give gift to another islander
            reputation += 1
            money -= 1
        elif action == 2: # Do nothing
            reward = -5
        elif action == 3: # Chat with another islander
            health -= 5
            reputation += 1
        elif action == 4: # Work
            health -= 10
            money += 2
        else: # Rob another islander
            health -= 20
            reputation -= 5
            money += 4
            reward = -1
        
        if health < 20 or reputation < 2 or money < 2:
            reward -= 2
            
        self.state = (health, reputation, money)

        done = bool(
            health < 0
            or reputation < 0
            or money < 0
        )
        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def reset(
        self
    ):
        self.state = (100.0, 10.0, 10.0)
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
