#!/usr/bin/env python3
import os
import sys
import argparse
sys.path.append('_model')
from agent import *

####### Parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Specifies which environment to run.', required=True)
args = parser.parse_args()

####### Defining Korali Problem

import korali
k = korali.Engine()
e = korali.Experiment()

### Defining results folder and loading previous results, if any

resultFolder = '_result_vracer/' 
e.loadState(resultFolder + '/latest');

### Initializing openAI Gym environment

initEnvironment(e, args.env)

### Defining Agent Configuration 

e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Updates Between Reward Rescaling"] = 200000
e["Solver"]["Episodes Per Generation"] = 1
e["Solver"]["Policy Distribution"] = "Normal"
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Learning Rate"] = 1e-4
e["Solver"]["L2 Regularization"]["Enabled"] = True
e["Solver"]["L2 Regularization"]["Importance"] = 1.0
e["Solver"]["Mini Batch Size"] = 256

### Defining the configuration of replay memory

e["Solver"]["Experience Replay"]["Start Size"] = 32768
e["Solver"]["Experience Replay"]["Maximum Size"] = 262144

### Configuring the Remember-and-Forget Experience Replay algorithm

e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5e-7
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3

### Configuring the neural network and its hidden layers

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Setting file output configuration

e["Solver"]["Termination Criteria"]["Max Experiences"] = 10e6
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 100
e["File Output"]["Path"] = resultFolder

### Running Experiment

k.run(e)
