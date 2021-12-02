//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include "Obstacles/SmartCylinder.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"

extern int _argc;
extern char **_argv;

void runEnvironmentVracer(korali::Sample &s);
void runEnvironmentMocmaes(korali::Sample &s);
void runEnvironmentCmaes(korali::Sample &s);
void setInitialConditions(SmartCylinder* agent, std::vector<double>& start, bool randomized);
bool isTerminal( SmartCylinder* agent, std::vector<double>& target );

// Helper functions
inline double distance(std::vector<double> x, std::vector<double> y) { return std::sqrt((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])); }
std::vector<double> logDivision(double start, double end, size_t nedges);
