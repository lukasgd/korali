// Select which environment to use
#include "_models/transportEnvironment/transportEnvironment.hpp"
#include "korali.hpp"

std::string _resultsPath;

int main(int argc, char *argv[])
{
  // Gathering actual arguments from MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }

  // Storing parameters
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 2;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine

  // Init CUP2D
  _environment = new Simulation(_argc, _argv);
  _environment->init();

  std::string resultsPath = "results_transport_cmaes/";

  // Creating Experiment
  auto e = korali::Experiment();
  e["Random Seed"] = 0xC0FEE;
  e["Problem"]["Type"] = "Optimization";
  
  auto found = e.loadState(resultsPath + std::string("latest"));
  if (found == true) printf("[Korali] Continuing execution from previous run...\n");

  // Configuring Experiment
  e["Problem"]["Objective Function"] = &runEnvironmentCmaes;

  const double maxForce = 1e-2;
  const size_t numVariables = 8;
  
  // Configuring CMA-ES parameters
  e["Solver"]["Type"] = "Optimizer/CMAES";
  e["Solver"]["Population Size"] = 4;
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-16;
  e["Solver"]["Termination Criteria"]["Max Generations"] = 5;
 
  // Setting up the variables
  for (size_t var = 0; var < numVariables; ++var)
  {
    e["Variables"][var]["Name"] = std::string("Force") + std::to_string(var);
    e["Variables"][var]["Lower Bound"] = 0.0;
    e["Variables"][var]["Upper Bound"] = +maxForce;
    e["Variables"][var]["Initial Standard Deviation"] = 0.3*maxForce/std::sqrt(numVariables);
  }

  ////// Setting Korali output configuration

  e["Console Output"]["Verbosity"] = "Detailed";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = resultsPath;

  ////// Running Experiment

  auto k = korali::Engine();

  // Configuring profiler output

  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = resultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 10;

  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Communicator"] = MPI_COMM_WORLD;

  k.run(e);
}