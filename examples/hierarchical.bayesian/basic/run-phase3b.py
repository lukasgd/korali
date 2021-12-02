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
sub.loadState('_setup/results_phase_1/000/latest')
psi.loadState('_setup/results_phase_2/latest')

# We need to redefine the subproblem's computational model
sub["Problem"]["Computational Model"] = lambda d: normal(N,d)

# Specifying reference data
data = getReferenceData("_setup/data/", 0)
N = len(data)
  
e["Problem"]["Type"] = "Hierarchical/Theta"
e["Problem"]["Sub Experiment"] = sub
e["Problem"]["Psi Experiment"] = psi

e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 1000
e["Solver"]["Burn In"] = 2
e["Solver"]["Max Chain Length"] = 1
e["Solver"]["Target Coefficient Of Variation"] = 0.6

e["Random Seed"] = 0xC0FFEE
e["Console Output"]["Verbosity"] = "Detailed"
e["File Output"]["Path"] = "_setup/results_phase_3b/"

# Starting Korali's Engine and running experiment
k = korali.Engine()
# k["Conduit"]["Type"] = "Concurrent"
# k["Conduit"]["Concurrent Jobs"] = 4
k.run(e)
