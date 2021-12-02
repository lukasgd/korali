#!/usr/bin/env python3

# Importing computational model
import sys
import os
import korali
sys.path.append('_setup/model')
from model import *

# Creating hierarchical Bayesian problem from previous two problems
e = korali.Experiment()
sub = korali.Experiment()
psi = korali.Experiment()

# Loading previous results
psi.loadState('_setup/results_phase_2/latest')
sub.loadState('_setup/results_phase_1/000/latest')

# Specifying reference data
x = getReferencePoints("_setup/data/", 0)

# We need to redefine the subproblem's computational model
sub["Problem"]["Computational Model"] = lambda d: logistic(x, d)

e["Problem"]["Type"] = "Hierarchical/Theta"
e["Problem"]["Psi Experiment"] = psi
e["Problem"]["Sub Experiment"] = sub

e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 1000
e["Solver"]["Termination Criteria"]["Max Generations"] = 30
e["Solver"]["Burn In"] = 1
e["Solver"]["Target Coefficient Of Variation"] = 0.6

e["Random Seed"] = 0xC0FFEE
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Path"] = "_setup/results_phase_3b/"

# Starting Korali's Engine and running experiment
k = korali.Engine()
k["Conduit"]["Type"] = "Concurrent"
k["Conduit"]["Concurrent Jobs"] = 4
k.run(e)
