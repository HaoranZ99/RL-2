"""
Simulation of a simple environment for the islanders to interact.
"""
import math
import re
from time import sleep
from typing import Optional, Union

import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding
from pyparsing import replaceWith

import logging

from torch import rand

class IslandEnv(gym.Env):
    """
    ### Description
    An agent is on a group of islands containing multiple ai's, the goal is to be able to live a little longer.

    ### Action Space
    The agent take a 1-element vector for actions.
    Actions:
    | Num  | Action              | Effects on States               |
    |------|---------------------|---------------------------------|
    |  0   | Eat                 | Money and Health                |
    |  1   | Send gift           | Money and Reputation            |
    |  2   | Idle                | /                               |
    |  3   | Chat                | Health and Reputation           |
    |  4   | Work                | Health and Money                |
    |  5   | Comments on Moments | Health and Reputation           |
    |  6   | Like on Moments     | Health and Reputation           |
    |  7   | Live room           | Health and Money and Reputation |
    |  8   | Play games          | Health and Money and Reputation |
    |  9   | Disco dancing       | Health and Money and Reputation |
    | 10   | Pray                | Health and Money                |

    ### Observation Space
    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
    | Num | Observation         | Min                  | Max                |
    |-----|---------------------|----------------------|--------------------|
    | 0   | Islander Health     | -Inf                 | Inf                |
    | 1   | Islander Reputation | -Inf                 | Inf                |
    | 2   | Islander Money      | -Inf                 | Inf                |

    ### Rewards
    Reward is 1 for every step taken except for action no.2.
    - If agent chooses to do nothing, reward = -5.
    - If agent chooses one action and if state is unhealthy, reward = -10.
    - If agent's health or reputation or money is too low, it receives continuous punishment, (reward -= 2).

    ### Starting state
    Islander's health is set to 100 and its money is set to 10 and its reputaiton is set to 10.

    ### Episode Termination
    The episode terminates of one of the following occurs:

    1. Islander's health is below 0.
    2. Islander's reputation is below 0.
    3. Islander's money is below 0.
    2. Episode length is greater than 500.   
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

        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32)

        self.state = None

    def _isDone(self, health=0, reputation=0, money=0):
        return bool(
            health < 0
            or reputation < 0
            or money < 0
        )
    
    def step(self, action):
        health, reputation, money = self.state
        
        reward = 1

        if action == 0: # Eat
            money -= 1
            if self._isDone(money=money):
                reward = -10
            else:
                health += 10
        elif action == 1: # Send gift
            money -= 1
            if self._isDone(money=money):
                reward = -10
            else:            
                reputation += 1
        elif action == 2: # Idle
            reward = -5
        elif action == 3: # Chat
            health -= 10
            if self._isDone(health=health):
                reward = -10
            else:
                reputation += 1
        elif action == 4: # Work
            health -= 10
            if self._isDone(health=health):
                reward = -10
            else:
                money += 1
        elif action == 5: # Comments on Moments
            health -= 5
            if self._isDone(health=health):
                reward = -10
            else:
                if np.random.uniform() > 0.7:
                    reputation += 2
                else:
                    reputation -= 2
        elif action == 6: # Like on Moments
            health -= 5
            if self._isDone(health=health):
                reward = -10
            else:
                reputation += 1
        elif action == 7: # Live room
            health -= 5
            money -= 1
            if self._isDone(health=health, money=money):
                reward = -10
            else:
                reputation += 1
        elif action == 8: # Play games
            health -= 10
            if self._isDone(health=health, money=money):
                reward = -10
            else:
                if np.random.uniform() > 0.5:
                    reputation += 2
                    money += 1
                else:
                    reputation -= 2
                    money -= 1
        elif action == 9: # Disco dancing
            health -= 10
            money -= 1
            if self._isDone(health=health, money=money):
                reward = -10
            else:
                reputation += 2
        else: # Pray
            health -= 5
            if self._isDone(health=health):
                reward = -10
            else:
                if np.random.uniform() > 0.7:
                    money += 1
                else:
                    health -= 20
        
        if health < 40 or reputation < 4 or money < 4:
            reward -= 10
            
        self.state = (health, reputation, money)

        '''
        Revise reward
        Introduce normal randomness.
        For moderate people, REWARD_OFFSIDE = {0 : 0.0, 1 : 0.0, 2 : 0.0, 3 : 0.0, 4 : 0.0, 5 : 0.0, 6 : 0.0, 7 : 0.0, 8 : 0.0, 9 : 0.0, 10 : 0.0}
        For conservative people, REWARD_OFFSIDE = {0 : 1.0, 1 : 1.0, 2 : 1.0, 3 : 1.0, 4 : 1.0, 5 : -1.0, 6 : 1.0, 7 : 1.0, 8 : -1.0, 9 : 1.0, 10 : -1.0}
        For aggressive people, REWARD_OFFSIDE = {0 : -1.0, 1 : -1.0, 2 : -1.0, 3 : -1.0, 4 : -1.0, 5 : 1.0, 6 : -1.0, 7 : -1.0, 8 : 1.0, 9 : -1.0, 10 : 1.0}
        '''
        REWARD_OFFSIDE = {0 : 0.0, 1 : 0.0, 2 : 0.0, 3 : 0.0, 4 : 0.0, 5 : 0.0, 6 : 0.0, 7 : 0.0, 8 : 0.0, 9 : 0.0, 10 : 0.0}
        mu = reward + REWARD_OFFSIDE.get(action)
        sigma = 2.0
        reward = np.random.normal(loc=mu, scale=sigma)

        done = bool(
            health < 0
            or reputation < 0
            or money < 0
        )

        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def reset(
        self
    ):
        self.state = (200.0, 20.0, 20.0)
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
