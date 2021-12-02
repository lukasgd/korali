import argparse
import sys
sys.path.append('_model')
from swarm import *
from plotter import *
import math
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--visualize', help='whether to plot the swarm or not', required=True, type=int)
    parser.add_argument('--numIndividuals', help='number of fish', required=True, type=int)
    parser.add_argument('--numTimesteps', help='number of timesteps to simulate', required=True, type=int)
    parser.add_argument('--numNearestNeighbours', help='number of nearest neighbours used for state/reward', required=True, type=int)

    args = vars(parser.parse_args())

    numIndividuals       = args["numIndividuals"]
    numTimeSteps         = args["numTimesteps"]
    numNearestNeighbours = args["numNearestNeighbours"]
    assert numIndividuals > numNearestNeighbours, print("numIndividuals must be bigger than numNearestNeighbours")
    
    sim  = swarm( numIndividuals, numNearestNeighbours )
    step = 0
    done = False
    action = [1,0,0]
    while (step < numTimeSteps) and not done:
        print("timestep {}/{}".format(step+1, numTimeSteps))
        # if enable, plot current configuration
        if args["visualize"]:
            Path("./figures").mkdir(parents=True, exist_ok=True)
            # fixed camera
            plotSwarm( sim, step )
            # camera following center of swarm
            # plotSwarmCentered( sim, step )

        # compute pair-wise distances and view-angles
        done = sim.preComputeStates()

        # update swimming directions
        for i in np.arange(sim.N):
            print("agent {}/{}".format(i+1, sim.N))
            # for Newton policy state is the directions to the nearest neighbours
            state  = sim.getState(i)
            # print("state:", state)
            # set action
            # action = sim.fishes[i].newtonPolicy( state )
            # print("action:", action)
            if math.isclose( np.linalg.norm(action),  1.0 ):
                sim.fishes[i].wishedDirection = action
            # get reward (Careful: assumes sim.state(i) was called before)
            reward = sim.getReward( i )
            print("reward:", reward)
            # rotation in wished direction
            sim.fishes[i].updateDirection()
            # update positions
            sim.fishes[i].updateLocation()

        step += 1
