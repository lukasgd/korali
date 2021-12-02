#!/usr/bin/env python3
from cartpole import *
import pdb
import numpy as np
######## Defining Environment Storage

cart = CartPole()
maxSteps = 500

def env(s):

 # Initializing environment and random seed
 sampleId = s["Sample Id"]
 launchId = s["Launch Id"]
 if s["Mode"] == "Training":
    envId = sampleId % 3
 else:
    envId = 0
 cart.reset(sampleId * 1024 + launchId)
 s["State"] =(cart.getState()).tolist()

 step = 0
 done = False

 while not done and step < maxSteps:
  
  s.update()
  
  # Performing the action
  done = cart.advance(s["Action"])
  
  # Getting Reward
  s["Reward"] = cart.getReward()
  
   
  # Storing New State
  s["State"] = (cart.getState()).tolist()
  
  # Advancing step counter
  step = step + 1

 # Setting finalization status
 if (cart.isOver()):
  s["Termination"] = "Terminal"
 else:
  s["Termination"] = "Truncated"