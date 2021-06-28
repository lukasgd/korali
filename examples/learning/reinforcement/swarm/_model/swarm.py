import random
import numpy as np
from itertools import product 
import time

from fish import *

class swarm:
    def __init__(self, N, numNN, alpha=360., speed=1., turningRate=360., cutOff=0.01, seed=42):
        assert alpha > 0., "wrong alpha"
        assert alpha <= 360., "wrong alpha"
        assert turningRate > 0., "wrong turningRate"
        assert turningRate <= 360., "wrong turningRate"
        # number of fish
        self.N = N
        # number of nearest neighbours
        self.numNearestNeighbours = numNN
        # field of perception
        self.alpha = alpha / 180. * np.pi
        # swimming speed 
        self.speed = speed
        # turning rate
        self.turningRate = turningRate / 180. * np.pi
        # cut off for "touching"
        self.cutOff = cutOff
        # create fish at random locations
        self.fishes = self.randomPlacementNoOverlap( seed )

    """ random placement on a grid """
    def randomPlacementNoOverlap(self, seed):
        # number of gridpoints per dimension for initial placement
        M = int( pow( self.N, 1/3 ) )
        V = M+1
        # grid spacing ~ min distance between fish
        dl = self.cutOff*10
        # maximal extent
        L = V*dl
        
        # generate random permutation of [1,..,V]x[1,..,V]x[1,..,V]
        perm = list(product(np.arange(0,V),repeat=3))
        assert self.N < len(perm), "More vertices required to generate random placement"
        random.Random( seed ).shuffle(perm)
    
        # place fish
        fishes = np.empty(shape=(self.N,), dtype=fish)
        for i in range(self.N):
          location = np.array([perm[i][0]*dl, perm[i][1]*dl, perm[i][2]*dl]) - L/2
          fishes[i] = fish(location, self.speed, self.turningRate)
        
        # return array of fish
        return fishes

    """ compute distance and angle matrix """
    def preComputeStatesNaive(self):
        # create containers for distances, angles and directions
        distances  = np.full( shape=(self.N,self.N), fill_value=np.inf, dtype=float)
        angles     = np.full( shape=(self.N,self.N), fill_value=np.inf, dtype=float)
        directions = np.zeros(shape=(self.N,self.N,3), dtype=float)
        # boolean indicating if two fish are touching
        terminal = False
        # iterate over grid and compute angle / distance matrix
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                if i != j:
                    # direction vector and current direction
                    u = self.fishes[j].location - self.fishes[i].location
                    v = self.fishes[i].curDirection
                    # set distance
                    distances[i,j] = np.linalg.norm(u)
                    # set direction
                    directions[i,j,:] = u
                    # set angle
                    cosAngle = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
                    angles[i,j] = np.arccos(cosAngle)
            # Termination state in case distance matrix has entries < cutoff
            if (distances[i,:] < self.fishes[i].sigmaPotential ).any():
                terminal = True

        self.distancesMat = distances
        self.anglesMat    = angles
        self.directionMat = directions

        return terminal

    """ compute distance and angle matrix """
    def preComputeStates(self):
        ## create containers for location and swimming directions and angles
        locations     = np.empty(shape=(self.N,3 ), dtype=float)
        curDirections = np.empty(shape=(self.N,3 ), dtype=float)

        ## fill matrix with locations / current swimming direction
        for i,fish in enumerate(self.fishes):
            locations[i,:]     = fish.location
            curDirections[i,:] = fish.curDirection
        # normalize swimming directions
        normalCurDirections = curDirections / np.linalg.norm( curDirections, axis=1 )[:, np.newaxis]

        ## create containers for direction, distance, and angle
        directions    = np.empty(shape=(self.N,self.N, 3), dtype=float)
        distances     = np.empty(shape=(self.N,self.N),    dtype=float)
        angles        = np.empty(shape=(self.N,self.N),    dtype=float)

        ## use numpy broadcasting to compute direction, distance, and angles
        directions    = locations[np.newaxis, :, :] - locations[:, np.newaxis, :]
        distances     = np.sqrt( np.einsum('ijk,ijk->ij', directions, directions) )
        # normalize direction
        normalDirections = directions / distances[:,:,np.newaxis]
        angles = np.arccos( np.einsum( 'ijk, ijk->ij', normalCurDirections[:,np.newaxis,:], normalDirections ) )
        
        ## set diagonals entries
        np.fill_diagonal( distances, np.inf )
        np.fill_diagonal( angles,    np.inf )

        ## fill values to class member variable
        self.directionMat = directions
        self.distancesMat = distances
        self.anglesMat    = angles
        
        # return if any two fish are closer then the cutOff
        return ( self.distancesMat < self.cutOff ).any()


    def getState( self, i ):
        # get array for agent i
        distances = self.distancesMat[i,:]
        angles    = self.anglesMat[i,:]
        directions= self.directionMat[i,:,:]
        # sort and select nearest neighbours
        idSorted = np.argsort( distances )
        idNearestNeighbours = idSorted[:self.numNearestNeighbours]
        self.distancesNearestNeighbours = distances[ idNearestNeighbours ]
        self.anglesNearestNeighbours    = angles[ idNearestNeighbours ]
        self.directionNearestNeighbours = directions[idNearestNeighbours,:]
        # the state is the distance (or direction?) and angle to the nearest neigbours
        return np.array([ self.distancesNearestNeighbours, self.anglesNearestNeighbours ]).flatten().tolist() # or np.array([ directionNearestNeighbours, anglesNearestNeighbours ]).flatten()

    def getReward( self, i ):
        # Careful: assumes sim.state(i) was called before
        return self.fishes[i].computeReward( self.distancesNearestNeighbours )

    ''' according to https://doi.org/10.1006/jtbi.2002.3065 '''
    def move(self):
        anglesMat, distancesMat = self.computeStates()
        for i in np.arange(self.N):
            deviation = anglesMat[i,:]
            distances = distancesMat[i,:]
            visible = abs(deviation) <= ( self.alpha / 2. )

            rRepell  = self.rRepulsion   * ( 1 + fishes[i].epsRepell  )
            rOrient  = self.rOrientation * ( 1 + fishes[i].epsOrient  ) 
            rAttract = self.rAttraction  * ( 1 + fishes[i].epsAttract )

            repellTargets  = self.fishes[(distances < rRepell)]
            orientTargets  = self.fishes[(distances >= rRepell) & (distances < rOrient) & visible]
            attractTargets = self.fishes[(distances >= rOrient) & (distances <= rAttract) & visible]
            self.fishes[i].computeDirection(repellTargets, orientTargets, attractTargets)

        for fish in self.fishes:
            fish.updateDirection()
            fish.updateLocation()

    ''' utility to compute polarisation (~alignement) '''
    def computePolarisation(self):
        polarisationVec = np.zeros(shape=(3,), dtype=float)
        for fish in self.fishes:
            polarisationVec += fish.curDirection
        polarisation = np.linalg.norm(polarisationVec) / self.N
        return polarisation

    ''' utility to compute center of swarm '''
    def computeCenter(self):
        center = np.zeros(shape=(3,), dtype=float)
        for fish in self.fishes:
            center += fish.location
        center /= self.N
        return center

    ''' utility to compute angular momentum (~rotation) '''
    def computeAngularMom(self):
        center = self.computeCenter()
        angularMomentumVec = np.zeros(shape=(3,), dtype=float)
        for fish in self.fishes:
            distance = fish.location-center
            distanceNormal = distance / np.linalg.norm(distance) 
            angularMomentumVecSingle = np.cross(distanceNormal,fish.curDirection)
            angularMomentumVec += angularMomentumVecSingle
        angularMomentum = np.linalg.norm(angularMomentumVec) / self.N
        return angularMomentum
