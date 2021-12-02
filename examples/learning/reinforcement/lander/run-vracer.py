#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Actions Between Policy Updates"] = 1

### Defining State variables

e["Variables"][0]["Name"] = "Position X"
e["Variables"][1]["Name"] = "Position Y"

### Defining Action variables 

e["Variables"][2]["Name"] = "Force X"
e["Variables"][2]["Type"] = "Action"
e["Variables"][2]["Lower Bound"] = -0.1
e["Variables"][2]["Upper Bound"] = +0.1
e["Variables"][2]["Initial Exploration Noise"] = 0.08

e["Variables"][3]["Name"] = "Force Y"
e["Variables"][3]["Type"] = "Action"
e["Variables"][3]["Lower Bound"] = +0.0
e["Variables"][3]["Upper Bound"] = +1.0
e["Variables"][3]["Initial Exploration Noise"] = 0.24

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Discount Factor"] = 0.99
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Mini Batch"]["Size"] = 32
e["Solver"]["Experience Replay"]["Start Size"] = 1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"
e["Solver"]["Time Sequence Length"] = 4

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Recurrent/GRU"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 16 

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = 50

### Setting file output configuration

e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
 
### Running Training Experiment

k.run(e)

### If this is test mode, we run a few test samples and check their reward

performTest = False
if len(sys.argv) == 2:
 if sys.argv[1] == '--test':
  performTest = True

if (performTest == False): exit(0)

e["Solver"]["Mode"] = "Testing"
e["Solver"]["Testing"]["Sample Ids"] = list(range(10))
e["File Output"]["Enabled"] = False

k.run(e)

averageTestReward = np.average(e["Solver"]["Testing"]["Reward"])
print("Average Reward: " + str(averageTestReward))
if (averageTestReward < -8.0):
 print("Landar example did not reach minimum testing average.")
 exit(-1)