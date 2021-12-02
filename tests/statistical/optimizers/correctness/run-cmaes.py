#!/usr/bin/env python3
import os
import sys
import korali

sys.path.append('./model')
sys.path.append('./helpers')

from model import *
from helpers import *

#################################################
# CMAES problem definition & run
#################################################

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.23246, 1e-4)

### Running With Diagonal Covariance Matrix

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 100
e["Solver"]["Diagonal Covariance"] = True

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.23246, 1e-4)

### Running With Mirrored Sampling

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 100
e["Solver"]["Mirrored Sampling"] = True

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

#k = korali.Engine()
k.run(e)

checkMin(e, 0.23246, 1e-4)

### Running With Different Mu Types

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 100
e["Solver"]["Mu Type"] = "Linear"

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.23246, 1e-4)

### Running With Different Mu Types

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 8
e["Solver"]["Termination Criteria"]["Max Generations"] = 100
e["Solver"]["Mu Type"] = "Logarithmic"

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.23246, 1e-4)

### Running With Different Mu Types

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 64
e["Solver"]["Termination Criteria"]["Max Generations"] = 100
e["Solver"]["Mu Type"] = "Proportional"

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.23246, 1e-3)

### Running With Different Mu Types

e = korali.Experiment()
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 64
e["Solver"]["Termination Criteria"]["Max Generations"] = 100
e["Solver"]["Mu Type"] = "Equal"

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.23246, 1e-3)

### Corner Case: Discrete with Mirrored Sampling

e = korali.Experiment()

# Selecting problem type
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Value"] = 1.0
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0
e["Variables"][0]["Granularity"] = 0.0001

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 64
e["Solver"]["Mirrored Sampling"] = True
e["Solver"]["Termination Criteria"]["Max Generations"] = 10

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkMin(e, 0.23246, 1e-3)

### Corner Case: Constraints that cannot be satisfied

e = korali.Experiment()

e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel
e["Problem"]["Constraints"] = [ constraint1 ]

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Value"] = 1.0
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 16
e["Solver"]["Viability Population Size"] = 2
e["Solver"]["Termination Criteria"]["Max Generations"] = 10

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkInfeasible(e, 10)

### Corner Case: Terminating on Max Infeasible Resamplings

e = korali.Experiment()

e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel
e["Problem"]["Constraints"] = [ constraint1 ]

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Value"] = 1.0
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 16
e["Solver"]["Viability Population Size"] = 2
e["Solver"]["Termination Criteria"]["Max Generations"] = 100
e["Solver"]["Termination Criteria"]["Max Infeasible Resamplings"] = 50

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

checkInfeasible(e, 50)

### Corner Case: Trigger Min StdDev Deviation Update Warning

e = korali.Experiment()

# Selecting problem type
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel

e["Variables"][0]["Name"] = "X"
e["Variables"][0]["Initial Value"] = 1.0
e["Variables"][0]["Lower Bound"] = -10.0
e["Variables"][0]["Upper Bound"] = +10.0
e["Variables"][0]["Minimum Standard Deviation Update"] = 1000.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 64
e["Solver"]["Termination Criteria"]["Max Generations"] = 10

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

### Corner Case: Trigger Max Covariance Matrix Corrections Warning

e = korali.Experiment()

e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = evalmodel
e["Problem"]["Constraints"] = [g1, g2, g3, g4]

for i in range(7):
  e["Variables"][i]["Name"] = "X" + str(i)
  e["Variables"][i]["Lower Bound"] = -10.0
  e["Variables"][i]["Upper Bound"] = +10.0

e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Is Sigma Bounded"] = True
e["Solver"]["Population Size"] = 32
e["Solver"]["Viability Population Size"] = 4
e["Solver"]["Max Covariance Matrix Corrections"] = 0
e["Solver"]["Termination Criteria"]["Max Value"] = -680.630057374402 - 1e-4
e["Solver"]["Termination Criteria"]["Max Generations"] = 500

e["Console Output"]["Frequency"] = 10
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Enabled"] = False
e["Random Seed"] = 1337

k = korali.Engine()
k.run(e)

