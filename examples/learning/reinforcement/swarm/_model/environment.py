from swarm import *
from pathlib import Path

def environment( args, s ):
    # set set parameters and initialize environment
    numIndividuals       = args["numIndividuals"]
    numTimesteps         = args["numTimesteps"]
    numNearestNeighbours = args["numNearestNeighbours"]
    sim = swarm( numIndividuals, numNearestNeighbours )

    # compute pair-wise distances and view-angles
    done = sim.preComputeStates()
    # set initial state
    states = []
    for i in np.arange(sim.N):
        # get state
        state = sim.getState( i )
        states.append( state )
    # print("states:", state)
    s["State"] = states

    ## run simulation
    step = 0
    if done: 
        print("Initial configuration is terminal state...")
    while (step < numTimesteps) and (not done):
        if args["visualize"]:
            Path("./_figures").mkdir(parents=True, exist_ok=True)
            # fixed camera
            # plotSwarm( sim, step )
            # camera following center of swarm
            plotSwarmCentered( sim, step )

        # Getting new action
        s.update()

        ## apply action, get reward and advance environment
        actions = s["Action"]
        # print("actions:", actions)
        for i in np.arange(sim.N):
            # compute wished direction based on action
            polarAngles = actions[i]
            x = np.cos(polarAngles[0])*np.sin(polarAngles[1])
            y = np.sin(polarAngles[0])*np.sin(polarAngles[1])
            z = np.cos(polarAngles[1])
            sim.fishes[i].wishedDirection = [ x, y, z ]
            # rotation in wished direction
            sim.fishes[i].updateDirection()
            # update positions
            sim.fishes[i].updateLocation()

        # compute pair-wise distances and view-angles
        done = sim.preComputeStates()
        # set state
        states  = []
        rewards = []
        for i in np.arange(sim.N):
            # get state
            state = sim.getState( i )
            states.append( state )
            # get reward
            reward = sim.getReward( i )
            if done:
                reward = -10.
            rewards.append(reward)
        # print("states:", states)
        s["State"] = states
        # print("rewards:", rewards)
        s["Reward"] = rewards

        step += 1

    # Setting termination status
    if done:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"