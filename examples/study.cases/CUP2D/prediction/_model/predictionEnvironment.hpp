//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include "Simulation.h"
#include "Utils/BufferedLogger.h"
#include <Cubism/BlockInfo.h>

using namespace cubism;

// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);
bool isTerminal();
