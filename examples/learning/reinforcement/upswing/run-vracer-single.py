#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from single_env import *

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position"
e["Variables"][1]["Name"] = "Angle 1"
e["Variables"][2]["Name"] = "Car Velocity"
e["Variables"][3]["Name"] = "Angular Velocity 1"
e["Variables"][4]["Name"] = "Height Proxy"

e["Variables"][5]["Name"] = "Force"
e["Variables"][5]["Type"] = "Action"
e["Variables"][5]["Lower Bound"] = -20.0
e["Variables"][5]["Upper Bound"] = +20.0
e["Variables"][5]["Initial Exploration Noise"] = 1.0

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Generations"] = 100

### If this is test mode, run only a couple generations
if len(sys.argv) == 2:
 if sys.argv[1] == '--test':
  e["Solver"]["Termination Criteria"]["Max Generations"] = 5
  
### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Mini Batch"]["Size"] = 256
e["Solver"]["Experience Replay"]["Start Size"] = 1000
e["Solver"]["Experience Replay"]["Maximum Size"] = 10000

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration

e["File Output"]["Enabled"] = False

### Running Experiment

k.run(e)
