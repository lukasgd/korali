import argparse
import sys
import math
sys.path.append('_model')
from environment import *

### Parsing arguments
parser = argparse.ArgumentParser()

parser.add_argument('--visualize', help='whether to plot the swarm or not, default is 0', required=False, type=int, default=0)
parser.add_argument('--numIndividuals', help='number of fish', required=True, type=int)
parser.add_argument('--numTimesteps', help='number of timesteps to simulate', required=True, type=int)
parser.add_argument('--numNearestNeighbours', help='number of nearest neighbours used for state/reward', required=True, type=int)

args = vars(parser.parse_args())

### check arguments
numIndividuals       = int(args["numIndividuals"])
numTimesteps         = int(args["numTimesteps"])
numNearestNeighbours = int(args["numNearestNeighbours"])
assert (numIndividuals > 0) & (numTimesteps > 0) & (numNearestNeighbours > 0) & (numIndividuals > numNearestNeighbours), print("invalid arguments: numTimesteps={}!>0, numIndividuals={}!>numNearestNeighbours={}!>0".format(numTimesteps, numIndividuals, numNearestNeighbours))

### Define Korali Problem
import korali
k = korali.Engine()
e = korali.Experiment()

### Define results folder and loading previous results, if any
resultFolder = '_result_vracer/'
found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Continuing execution from previous run...\n");

### Define Problem Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda x : environment( args, x )
e["Problem"]["Agents Per Environment"] = numIndividuals

### Define Agent Configuration 
e["Solver"]["Type"] = "Agent / Continuous / VRACER"
e["Solver"]["Mode"] = "Training"
e["Solver"]["Episodes Per Generation"] = 10
e["Solver"]["Experiences Between Policy Updates"] = 1
e["Solver"]["Learning Rate"] = 0.0001
e["Solver"]["Discount Factor"] = 0.995
e["Solver"]["Mini Batch"]["Size"] = 256

### Define Variables
# States (distance and angle to nearest neighbours)
for i in range(numNearestNeighbours):
  e["Variables"][i]["Name"] = "Distance " + str(i)
  e["Variables"][i]["Type"] = "State"
  e["Variables"][i+numNearestNeighbours]["Name"] = "Angle " + str(i)
  e["Variables"][i+numNearestNeighbours]["Type"] = "State"

# Actions (angles for spherical coordinates)
e["Variables"][2*numNearestNeighbours]["Name"] = "Phi"
e["Variables"][2*numNearestNeighbours]["Type"] = "Action"
e["Variables"][2*numNearestNeighbours]["Lower Bound"] = 0
e["Variables"][2*numNearestNeighbours]["Upper Bound"] = 2*np.pi
e["Variables"][2*numNearestNeighbours]["Initial Exploration Noise"] = np.pi

e["Variables"][2*numNearestNeighbours+1]["Name"] = "Theta"
e["Variables"][2*numNearestNeighbours+1]["Type"] = "Action"
e["Variables"][2*numNearestNeighbours+1]["Lower Bound"] = 0
e["Variables"][2*numNearestNeighbours+1]["Upper Bound"] = np.pi
e["Variables"][2*numNearestNeighbours+1]["Initial Exploration Noise"] = np.pi/2

### Set Experience Replay, REFER and policy settings
e["Solver"]["Experience Replay"]["Start Size"] = 10000
e["Solver"]["Experience Replay"]["Maximum Size"] = 100000
e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8
e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0
e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3
e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1

e["Solver"]["Policy"]["Distribution"] = "Squashed Normal"
e["Solver"]["State Rescaling"]["Enabled"] = True
e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True
# e["Solver"]["Reward"]["Outbound Penalization"]["Enabled"] = True
# e["Solver"]["Reward"]["Outbound Penalization"]["Factor"] = 0.5
  
### Configure the neural network and its hidden layers
e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]['Neural Network']['Optimizer'] = "Adam"
e["Solver"]["L2 Regularization"]["Enabled"] = True
e["Solver"]["L2 Regularization"]["Importance"] = 1.0

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

### Set file output configuration
e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e6
e["Solver"]["Experience Replay"]["Serialize"] = True
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 10
e["File Output"]["Path"] = resultFolder


### Run Experiment
k.run(e)
