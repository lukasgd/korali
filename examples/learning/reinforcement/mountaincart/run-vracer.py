#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--distribution',
    help='Policy Distribution',
    type=str,
    default='Normal',
    required=False)
parser.add_argument(
    '--maxExperiences',
    help='Number of experiences to collect.',
    type=int,
    default=1e6,
    required=False)


print("Running Mountaincart example with arguments:")
args = parser.parse_args()
print(args)

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining the Cartpole problem's configuration

e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = env
e["Problem"]["Actions Between Policy Updates"] = 1

e["Variables"][0]["Name"] = "Cart Position X"
e["Variables"][0]["Type"] = "State"

e["Variables"][1]["Name"] = "Cart Position Y"
e["Variables"][1]["Type"] = "State"

e["Variables"][2]["Name"] = "Cart Velocity X"
e["Variables"][2]["Type"] = "State"

e["Variables"][3]["Name"] = "Cart Velocity Y"
e["Variables"][3]["Type"] = "State"

e["Variables"][4]["Name"] = "Cart Acceleration X"
e["Variables"][4]["Type"] = "State"

e["Variables"][5]["Name"] = "Cart Acceleration Y"
e["Variables"][5]["Type"] = "State"

e["Variables"][6]["Name"] = "Force"
e["Variables"][6]["Type"] = "Action"
e["Variables"][6]["Lower Bound"] = -1.0
e["Variables"][6]["Upper Bound"] = +1.0
e["Variables"][6]["Initial Exploration Noise"] = 0.3

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Episodes Per Generation"] = 1

e["Solver"]["Experience Replay"]["Start Size"] = 4096
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144

e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Mini Batch"]["Size"] = 256

e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
e["Solver"]["L2 Regularization"]["Enabled"] = False

### Configuring the neural network and its hidden layers

e["Solver"]["Policy"]["Distribution"] = args.distribution
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adam"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Defining Termination Criteria

e["Solver"]["Termination Criteria"]["Max Experiences"] = args.maxExperiences

### Setting file output configuration

e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 500
e["File Output"]["Path"] = '_korali_results_{}'.format(e["Solver"]["Policy"]["Distribution"].replace(' ','_'))

### Running Experiment

k.run(e)