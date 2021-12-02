import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import vonmises
from scipy.stats import truncnorm

from plotter import *

# parameters for truncated gaussians (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html), taken from https://www.sciencedirect.com/science/article/pii/0304380094900132 
observedMean  = np.array([ 0.7, 1.0, 1.3 ])
observedSigma = np.array([ 0.3, 0.4, 0.5 ])
lowerBound = 0
upperBound = np.inf
observedA = (lowerBound-observedMean)/observedSigma
observedB = (upperBound-observedMean)/observedSigma


class fish:
    def __init__(self, location, individualStd=0.05, speed=1, maxAngle=30./180.*np.pi, eqDistance=0.1, potentialStrength=100, potential="Observed" ):
        self.location = location
        self.curDirection = self.randUnitDirection()
        self.wishedDirection = self.curDirection
        # individual variation
        self.individualStd = individualStd
        individualNoise = np.zeros(4) #np.random.normal(0.0, self.individualStd, 4)
        # motion parameters
        self.speed       = speed    * ( 1 + individualNoise[0] )
        self.maxAngle    = maxAngle * ( 1 + individualNoise[1] )
        self.dt          = 0.1
        self.sigmaMotion = 0.1
        # potential
        self.potential = potential
        ## parameters for potentials
        # max value of reward
        self.epsilon        = potentialStrength * ( 1 + individualNoise[2] )
        # distance below which reward becomes penality
        self.sigmaPotential = eqDistance        * ( 1 + individualNoise[3] )

    ''' get uniform random unit vector on sphere '''      
    def randUnitDirection(self):
        vec = np.random.normal(0.,1.,3)
        mag = np.linalg.norm(vec)
        return vec/mag

    ''' according to https://doi.org/10.1006/jtbi.2002.3065 and/or https://hal.archives-ouvertes.fr/hal-00167590 '''
    def computeDirection(self, repellTargets, orientTargets, attractTargets):
        newWishedDirection = np.zeros(3)
        # zone of repulsion - highest priority
        if repellTargets.size > 0:
            for fish in repellTargets:
                diff = fish.location - self.location
                assert np.linalg.norm(diff) > 1e-12, print(diff, "are you satisfying speed*dt<=rRepulsion?")
                assert np.linalg.norm(diff) < 1e12,  print(diff)
                newWishedDirection -= diff/np.linalg.norm(diff)
        else:
            orientDirect = np.zeros(3)
            attractDirect = np.zeros(3)
            # zone of orientation
            if orientTargets.size > 0:
              for fish in orientTargets:
                  orientDirect += fish.curDirection/np.linalg.norm(fish.curDirection)
            # zone of attraction
            if attractTargets.size > 0:
              for fish in attractTargets:
                  diff = fish.location - self.location
                  attractDirect += diff/np.linalg.norm(diff)
            
            newWishedDirection = orientDirect+attractDirect
        
        if np.linalg.norm(newWishedDirection) < 1e-12:
          newWishedDirection = self.curDirection
        
        ## stochastic effect, replicates "spherically wrapped Gaussian distribution"
        # get random unit direction orthogonal to newWishedDirection
        randVector = self.randUnitDirection()
        rotVector = np.cross(newWishedDirection,randVector)
        while np.isclose(np.linalg.norm(rotVector), 0.0):
            randVector = self.randUnitDirection()
            rotVector = np.cross(newWishedDirection,randVector)
        rotVector /= np.linalg.norm(rotVector)
        # compute random angle from wrapped Gaussian ~ van Mises distribution
        randAngle = vonmises.rvs(1/self.sigma**2)
        # create rotation
        rotVector *= randAngle
        r = Rotation.from_rotvec(rotVector)
        # apply rotation
        self.wishedDirection = r.apply(newWishedDirection)


    ''' rotate direction of the swimmer ''' 
    def updateDirection(self):
        u = self.curDirection
        v = self.wishedDirection
        assert np.isclose( np.linalg.norm(u), 1.0 ), "Current direction {} not normalized".format(u)
        assert np.isclose( np.linalg.norm(v), 1.0 ), "Wished direction {} not normalized".format(v)

        # numerical safe computation of cos and angle
        cosAngle = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
        cosAngle = np.clip(cosAngle, -1, 1)
        angle    = np.arccos(cosAngle)
        if angle < self.maxAngle:
            self.curDirection = self.wishedDirection
        # handle antiparallel case
        elif np.isclose(angle, np.pi):
            randVector = self.randUnitDirection()
            rotVector = np.cross(self.curDirection,randVector)
            while np.isclose(np.linalg.norm(rotVector), 0.0):
                randVector = self.randUnitDirection()
                rotVector = np.cross(self.curDirection,randVector)
            rotVector /= np.linalg.norm(rotVector)
            rotVector *= self.maxAngle
            r = Rotation.from_rotvec(rotVector)
            self.curDirection = r.apply(self.curDirection)
        else:
            rotVector = np.cross(self.curDirection, self.wishedDirection)
            assert np.linalg.norm(rotVector) > 0, "Rotation vector {} from current {} and wished direction {} with angle {} is zero".format(rotVector, self.curDirection, self.wishedDirection, cosAngle)
            rotVector /= np.linalg.norm(rotVector)
            rotVector *= self.maxAngle
            r = Rotation.from_rotvec(rotVector)
            self.curDirection = r.apply(self.curDirection)
        
        # normalize
        self.curDirection /= np.linalg.norm(self.curDirection)

    ''' update the direction according to x += vt ''' 
    def updateLocation(self):
        self.location += self.speed*self.dt*self.curDirection

    ''' reward assumes pair-wise potentials ''' 
    def computeReward(self, nearestNeighbourDistance ):
        reward = 0.0
        for i,r in enumerate(nearestNeighbourDistance):
            # Lennard-Jones potential
            if self.potential == "Lennard-Jones":
                x = self.sigmaPotential / r
                reward -= 4*self.epsilon*( x**12 - x**6 )
            # Harmonic potential
            elif self.potential == "Harmonic":
                reward += self.epsilon - 4*self.epsilon/self.sigmaPotential**2*(156/2**(7/3)-42/2**(4/3))*(r-2**(1/6)*self.sigmaPotential)**2
            # Observations (https://www.sciencedirect.com/science/article/pii/0304380094900132)
            elif self.potential == "Observed":
                if i>2:
                    assert 0, print("The 'Observed' reward only supports up to 3 nearest Neighbours")
                # rTest = np.linspace(-10,10,1001)
                # plt.plot(rTest, truncnorm.pdf(rTest, a=observedA[i], b=observedB[i], loc=observedMean[i], scale=observedSigma[i]))
                reward += truncnorm.pdf(r, a=observedA[i], b=observedB[i], loc=observedMean[i], scale=observedSigma[i])
            else:
                assert 0, print("Please chose a pair-potential that is implemented")
        # plt.show()
        # print(nearestNeighbourDistance, reward)
        return reward

    ''' newton policy computes direction as gradient of potential ''' 
    def newtonPolicy(self, nearestNeighbourDirections ):
        action = np.zeros(3)
        for direction in nearestNeighbourDirections:
            r = np.linalg.norm(direction)
            # Lennard-Jones potential
            if self.potential == "Lennard-Jones":
                x = self.sigmaPotential / r
                action -= 4*self.epsilon*( -12*x**12/r + 6*x**6/r )*direction/r
            # Harmonic potential
            elif self.potential == "Harmonic":
                action += 4*self.epsilon/self.sigmaPotential**2*(156/2**(7/3)-42/2**(4/3))*(r-2**(1/6)*self.sigmaPotential)*direction/r
            elif self.potential == "Observed":
                assert 0, print("please do first implement the policy for the 'Observed' reward")
            else:
                assert 0, print("Please chose a pair-potential that is implemented")
        action = action / np.linalg.norm(action)
        return action

