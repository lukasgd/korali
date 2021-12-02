import argparse
import sys
import math
sys.path.append('_model')
from environment import *

### Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--visualize', help='whether to plot the swarm or not', required=True, type=int)
parser.add_argument('--numIndividuals', help='number of fish', required=True, type=int)
parser.add_argument('--numTimesteps', help='number of timesteps to simulate', required=True, type=int)
parser.add_argument('--numNearestNeighbours', help='number of nearest neighbours used for state/reward', required=True, type=int)

args = vars(parser.parse_args())

### Check given arguments
numIndividuals       = args["numIndividuals"]
numTimesteps         = args["numTimesteps"]
numNearestNeighbours = args["numNearestNeighbours"]
assert (numIndividuals > 0) & (numTimesteps > 0) & (numNearestNeighbours > 0) & (numIndividuals > numNearestNeighbours), print("invalid arguments: numTimesteps={}!>0, numIndividuals={}!>numNearestNeighbours={}!>0".format(numTimesteps, numIndividuals, numNearestNeighbours))

### Define Korali Problem
import korali
k = korali.Engine()
e = korali.Experiment()

### Define results folder and loading results
resultFolder = '_result_vracer/'
found = e.loadState(resultFolder + '/latest')
if found == True:
	print("[Korali] Evaluating previous run...\n");
else:
	print("[Korali] Error: could not find results in folder: " + resultFolder)
	exit(-1)

### Define Configuration
e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
e["Problem"]["Environment Function"] = lambda x : environment( args, x )
e["Solver"]["Mode"] = "Testing"
e["Solver"]["Testing"]["Sample Ids"] = [ 42 ] 
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = True
e["File Output"]["Frequency"] = 1
e["File Output"]["Path"] = resultFolder

### Run Experiment
k.run(e)