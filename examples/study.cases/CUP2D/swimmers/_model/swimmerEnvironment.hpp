//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include "Obstacles/StefanFish.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"

// #define ID
// #define MULTITASK
#define SWARM
// #define SINGLE
// #define WATERTURBINE

// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);
void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining);
bool isTerminal(StefanFish *agent, size_t nAgents);
