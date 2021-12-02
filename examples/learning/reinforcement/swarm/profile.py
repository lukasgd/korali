import argparse
import sys
sys.path.append('_model')
from swarm import *
from plotter import *
import math
from pathlib import Path
from line_profiler import LineProfiler

if __name__ == '__main__':
    numIndividuals       = 100
    numNearestNeighbours = 3
    
    sim  = swarm( numIndividuals, numNearestNeighbours )
    
    lp = LineProfiler()
    # naive version using for loops
    lp(sim.preComputeStatesNaive)()
    distance1 = sim.distancesMat
    directionMat1 = sim.directionMat
    anglesMat1 = sim.anglesMat

    # numpy version
    lp(sim.preComputeStates)()
    distance2 = sim.distancesMat
    directionMat2 = sim.directionMat
    anglesMat2 = sim.anglesMat

    # print profiling
    lp.print_stats()

    # make sure results are the same
    assert np.allclose(distance1, distance2)
    assert np.allclose(directionMat1, directionMat2)
    assert np.allclose(anglesMat1, anglesMat2)
