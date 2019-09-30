# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:00:42 2019

@author: Romain
"""

import gym
import retro

env = retro.make('Airstriker-Genesis','Level1')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

