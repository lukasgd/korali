#include "korali.hpp"
#include <algorithm>
#include <random>
#include "Obstacles/Windmill.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"

extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);
void setInitialConditions(Windmill* agent, double init_angle, bool randomized);