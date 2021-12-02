//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "transportEnvironment.hpp"
#include "spline.h"
#include <chrono>
#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

void runEnvironmentVracer(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId);
  std::filesystem::create_directories(resDir);

  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "a", stdout);
  if (logFile == NULL)
  {
    printf("Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Creating simulation environment
  Simulation *_environment = new Simulation(_argc, _argv);
  _environment->init();

  // Obtaining agent
  SmartCylinder* agent = dynamic_cast<SmartCylinder *>(_environment->getShapes()[0]);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Reseting environment and setting initial conditions
  std::vector<double> start{1., 2.};
  setInitialConditions(agent, start, s["Mode"] == "Training");

  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Set target 
  std::vector<double> target{3., 2.};

  // Setting initial state
  auto state = agent->state( target );
  s["State"] = state;

  // Setting initial time and step conditions
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  size_t curStep = 0;  // current Step

  // Setting maximum number of steps before truncation
  size_t maxSteps = 15e3;

  // Setting timestep between actions
  double dtact = 1e-1;

  // Starting main environment loop
  bool done = false;
  while (done == false && curStep < maxSteps)
  {
    // Getting initial time
    auto beginTime = std::chrono::steady_clock::now(); // Profiling

    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Setting action
    if (action.size() == 2)
    {
      agent->act( action );
    }
    else if (action.size() == 3)
    {
      std::vector<double> force = { action[0], action[1] };
      agent->act( force );
      agent->torque = action[2];
    }
    else
    {
      fprintf(stderr, "Action must be of size 2 or 3.\n");
      exit(-1);
    }

    // Run the simulation until next action is required
    tNextAct += dtact;
    while ( t < tNextAct )
    {
      // Calculate simulation timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtact);
      t += dt;

      // advance simulation and check if the simulation ends
      if (_environment->advance(dt))
      {
        fprintf(stderr, "Environment finished..\n");
        exit(-1);
      }
      // Re-check if simulation is done.
      done = isTerminal( agent, target );
    }

    double reward = agent->reward( target );

    // Getting ending time
    auto endTime = std::chrono::steady_clock::now(); // Profiling
    double actionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() / 1.0e+9;

    // Printing Information:
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ %.6f", state[0]);
    for (size_t i = 1; i < state.size(); i++) printf(", %.6f", state[i]);
    printf("]\n");
    printf("[Korali] Force: [ %.6f, %.6f ]\n", action[0], action[1]);
    if (action.size() == 3)
        printf("[Korali] Torque: %.6f\n", action[2]);
    printf("[Korali] Reward: %.6f\n", reward);
    printf("[Korali] Terminal?: %d\n", done);
    printf("[Korali] Time: %.3fs\n", actionTime);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);

    // Obtaining new agent state
    state = agent->state( target );

    // Storing reward
    s["Reward"] = reward;

    // Storing new state
    s["State"] = state;

    // Advancing to next step
    curStep++;
  }

  // Flush CUP logger
  logger.flush();

  // delete simulation class
  delete _environment;

  // Setting finalization status
  if (done == true)
    s["Termination"] = "Terminal";
  else
    s["Termination"] = "Truncated";

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

void runEnvironmentMocmaes(korali::Sample &s)
{
  size_t sampleId = s["Sample Id"];

  // Creating results directory
  std::string baseDir = "_log_transport_mocmaes/";
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", baseDir.c_str(), sampleId);
  std::filesystem::create_directories(resDir); 
  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "a", stdout);
  if (logFile == NULL)
  {
    printf("Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Creating simulation environment
  Simulation *_environment = new Simulation(_argc, _argv);
  _environment->init();

  // Environment Setup
  double startX = 1.0;
  double endX = 3.0;
  double height = 2.0;
  double dtact = 1e-1;
  // environment (flow field) dump frequency
  _environment->sim.dumpTime = 0.0;
  // Set target
  std::vector<double> target{endX, height};

  // Constraints
  size_t maxSteps = 1e3;
  double maxEnergy = 1e-1;

  // Setting time/energy/counter to zero
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  double energy = 0.0; // Total energy
  size_t curStep = 0;  // Current Step

  // Obtaining agent
  SmartCylinder* agent = dynamic_cast<SmartCylinder *>(_environment->getShapes()[0]);

  // Resetting environment and setting initial conditions
  std::vector<double> start{startX, height};
  setInitialConditions(agent, start, false);

  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Get parameterisation of force from MOCMA
  std::vector<double> params = s["Parameters"];
  size_t numParams = params.size();

  double* centerArr = agent->center;
  std::vector<double> currentPos(centerArr, centerArr+2);
  
  //std::vector<double> vertices = logDivision(startX, endX, numParams+1);
  std::vector<double> edges(numParams, 0.0);
  for(size_t i = 0; i < numParams; ++i) edges[i] = startX + i*(endX-startX)/(float)(numParams-1.);

  // Natural cubic spline (C^2) with natural boundary conditions (f''=0)
  tk::spline forceSpline(edges, params);

  // Init counting variables
  double distToTarget = distance(currentPos, target);

  // Starting main environment loop
  bool done = false;
  std::vector<double> action(2, 0.0);
  while (done == false && curStep < maxSteps)
  {
    centerArr = agent->center;
    currentPos[0] = centerArr[0];
    currentPos[1] = centerArr[1];

    double force = std::abs(forceSpline(currentPos[0])); // std::abs because spline evaluation may be negative

    if (distToTarget > 0.)
    {
      // Split force in x & y component
      action[0] = force*(target[0]-currentPos[0])/distToTarget;
      action[1] = force*(target[1]-currentPos[1])/distToTarget;
    }
    else
    {
      // Safe split in case of close distance
      action[0] = force*(target[0]-currentPos[0]);
      action[1] = force*(target[1]-currentPos[1]);
    }

    // Setting action
    agent->act( action );

    // Run the simulation until next action is required
    tNextAct += dtact;
    while ( t < tNextAct )
    {
      // Calculate simulation timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtact);
      t += dt;

      // advance simulation and check if the simulation ends
      if (_environment->advance(dt))
      {
        fprintf(stderr, "Environment finished\n");
        exit(-1);
      }
    }
 
    // Update distance
    distToTarget = distance(currentPos, target);
    energy = agent->energy;

    // Checking termination
    done = (currentPos[0] >= endX) || (curStep >= maxSteps) || (energy >= maxEnergy);
 
    curStep++;
    
    // Printing Information:
    printf("[Korali] Sample %lu, Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ %.6f, %.6f ]\n", currentPos[0], currentPos[1]);
    printf("[Korali] Force: [ %.6f, %.6f ]\n", action[0], action[1]);
    printf("[Korali] Energy %f, Distance %f, Terminal?: %d\n", energy, distToTarget, done);
    printf("[Korali] Time: %.3fs\n", t);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);
  }
  
  // Penalization for not reaching target
  if (currentPos[0] < endX)
  {
    printf("Target not reached, penalizing objectives..\n");
    t += (endX-currentPos[0])*1e9;
    energy += (endX-currentPos[0])*1e9;
  }
  if (energy > maxEnergy)
  {
    printf("Max energy violated (%f), penalizing objectives..\n", maxEnergy);
    t += (energy-maxEnergy)*1e9;
    energy += (energy-maxEnergy)*1e9;
  }
  
  // Setting Objectives
  std::vector<double> objectives = { -t, -energy };
  printf("Objectives: %f (time), %f (energy) (total steps %zu) \n", t, energy, curStep);
  s["F(x)"] = objectives;

  // Flush CUP logger
  logger.flush();

  // delete simulation class
  delete _environment;

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

void runEnvironmentCmaes(korali::Sample& s)
{
  size_t sampleId = s["Sample Id"];

  // Creating results directory
  std::string baseDir = "_log_transport_cmaes/";
  char resDir[64];
  sprintf(resDir, "%s/sample%08lu", baseDir.c_str(), sampleId);
  std::filesystem::create_directories(resDir); 
  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "a", stdout);
  if (logFile == NULL)
  {
    printf("Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Creating simulation environment
  Simulation *_environment = new Simulation(_argc, _argv);
  _environment->init();

  // Environment Setup
  double startX = 1.0;
  double endX = 3.0;
  double height = 2.0;
  double dtact = 1e-1;
  // environment (flow field) dump frequency
  _environment->sim.dumpTime = 0.0;

  // Constraints
  size_t maxSteps = 1e3;
  double maxEnergy = 1e-1;

  // Setting time/energy/counter to zero
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  double energy = 0.0; // Total energy
  size_t curStep = 0;  // Current Step

  // Obtaining agent
  SmartCylinder* agent = dynamic_cast<SmartCylinder *>(_environment->getShapes()[0]);

  // Reseting environment and setting initial conditions
  std::vector<double> start{startX, height};
  setInitialConditions(agent, start, false);

  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Get parameterisation of force from CMA
  std::vector<double> params = s["Parameters"];
  const double a = params[0];
  const double b = params[1];
  const double c = params[2];
  const double d = params[3];
  const double e = params[4];

  // Force applied
  const double maxForce = 1e-2;
  
  // Safety intervall before boundary (eps + radius)
  const double deps = 3e-1;
  
  // Starting main environment loop
  bool done = false;
  std::vector<double> action(2, 0.0);
  double* centerArr = agent->center;
  std::vector<double> currentPos(centerArr, centerArr+2);
  while (done == false && curStep < maxSteps)
  {
    centerArr = agent->center;
    currentPos[0] = centerArr[0];
    currentPos[1] = centerArr[1];
 
    // Compute force vector
    const double x = currentPos[0];
    double forcex = 1.;
    double forcey = (d*x+e)*(0.5*a/std::sqrt(x)+b+2.*c*x)*std::cos(a*std::sqrt(x)+x*b+c*x*x)+d*std::sin(a*std::sqrt(x)+x*b+c*x*x);

    // Force vector normalization
    const double invFvecLength = 1./std::sqrt(forcey*forcey+forcex*forcex);
    forcey *= invFvecLength;
    forcex *= invFvecLength;

    // Split force in x & y component
    action[0] = forcex*maxForce;
    action[1] = forcey*maxForce;

    // Setting action
    agent->act( action );

    // Run the simulation until next action is required
    tNextAct += dtact;
    while ( t < tNextAct )
    {
      // Calculate simulation timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtact);
      t += dt;

      // advance simulation and check if the simulation ends
      if (_environment->advance(dt))
      {
        fprintf(stderr, "Environment finished\n");
        exit(-1);
      }
    }

    // get energy used
    energy = agent->energy;

    // Checkting termination
    done = (currentPos[0] >= endX) || (curStep >= maxSteps);
 
    curStep++;
    
    // Printing Information:
    printf("[Korali] Sample %lu, Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ %.6f, %.6f ]\n", currentPos[0], currentPos[1]);
    printf("[Korali] Force: [ %.6f, %.6f ]\n", action[0], action[1]);
    printf("[Korali] Energy %f, Terminal?: %d\n", energy, done);
    printf("[Korali] Time: %.3fs\n", t);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);
 
    if (currentPos[0] < deps) 
    {
        done = true; // cylinder approaching left bound
        printf("[Korali] Terminating, Cylinder approaching left bound\n");
    }
    if (currentPos[1] > 4.0 - deps)
    {
        done = true; // cylinder approaching upper bound
        printf("[Korali] Terminating, Cylinder approaching upper bound\n");
    }
    else if (currentPos[1] < deps)
    {
        done = true; // cylinder approaching lower bound
        printf("[Korali] Terminating, Cylinder approaching lower bound\n");
    }

  }
  
  // Penalization for not reaching target
  if (currentPos[0] < endX)
  {
    printf("Target not reached, penalizing objectives..\n");
    t += (endX-currentPos[0])*1e9;
  }
  
  // Setting Objectives
  printf("Objectives: %f (time), %f (energy) (total steps %zu) \n", t, energy, curStep);
  s["F(x)"] = -t;

  // Flush CUP logger
  logger.flush();

  // delete simulation class
  delete _environment;

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

void setInitialConditions(SmartCylinder* agent, std::vector<double>& start, bool randomized)
{
  // Initial fixed conditions
  double locationX = start[0];
  double locationY = start[1];

  // or with noise
  if (randomized)
  {
    std::uniform_real_distribution<double> dis(-0.01, 0.01);

    double distX = dis(_randomGenerator);
    double distY = dis(_randomGenerator);

    locationX += distX;
    locationY += distY;
  }

  printf("[Korali] Initial Conditions:\n");
  printf("[Korali] locationX: %f\n", locationX);
  printf("[Korali] locationY: %f\n", locationY);

  // Setting initial position and orientation for the fish
  double C[2] = { locationX, locationY};
  agent->setCenterOfMass(C);

  // Reset energy
  agent->energy = 0.;
}

bool isTerminal(SmartCylinder* agent, std::vector<double>& target )
{
  const double dX = (agent->center[0] - target[0]);
  const double dY = (agent->center[1] - target[1]);

  const double dTarget = std::sqrt(dX*dX+dY*dY);

  bool terminal = false;
  if ( dTarget < 1e-1 ) terminal = true;

  return terminal;
}

std::vector<double> logDivision(double start, double end, size_t nvertices)
{
    std::vector<double> vertices(nvertices, 0.0);
    for(size_t idx = 0; idx < nvertices; ++idx)
    {
        vertices[idx] = std::exp((double) idx / (double) (nvertices-1.0) * std::log(end-start+1.0)) - 1.0 + start;
  printf("v %zu %lf\n", idx, vertices[idx]);
    }
    return vertices;
}
