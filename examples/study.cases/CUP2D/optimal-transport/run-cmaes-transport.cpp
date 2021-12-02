// Select which environment to use
#include "_model/transportEnvironment.hpp"
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
 
  // Setting results path
  std::string resultsPath = "_results_transport_cmaes/";

  // Creating Experiment
  auto e = korali::Experiment();

  // Check if existing results are there and continuing them
  auto found = e.loadState(resultsPath + std::string("latest"));
  if (found == true) printf("[Korali] Continuing execution from previous run...\n");

  // Configuring Experiment
  e["Random Seed"] = 0xC0FEE;
  e["Problem"]["Type"] = "Optimization";
  e["Problem"]["Objective Function"] = &runEnvironmentCmaes;
  
  // Configuring CMA-ES parameters
  e["Solver"]["Type"] = "Optimizer/CMAES";
  e["Solver"]["Population Size"] = 32;
  e["Solver"]["Mu Value"] = 16;
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-16;
  e["Solver"]["Termination Criteria"]["Max Generations"] = 500;
 
  // Setting up the variables
  e["Variables"][0]["Name"] = "a";
  e["Variables"][0]["Lower Bound"] = 0.;
  e["Variables"][0]["Upper Bound"] = 100.;

  e["Variables"][1]["Name"] = "b";
  e["Variables"][1]["Lower Bound"] = 0.;
  e["Variables"][1]["Upper Bound"] = 100.;

  e["Variables"][2]["Name"] = "c";
  e["Variables"][2]["Lower Bound"] = 0.;
  e["Variables"][2]["Upper Bound"] = 100.;
 
  e["Variables"][3]["Name"] = "d";
  e["Variables"][3]["Lower Bound"] = -10.;
  e["Variables"][3]["Upper Bound"] = 10.;

  e["Variables"][4]["Name"] = "e";
  e["Variables"][4]["Lower Bound"] = -10.;
  e["Variables"][4]["Upper Bound"] = 10.;

  ////// Setting Korali output configuration
  e["Console Output"]["Verbosity"] = "Detailed";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = resultsPath;

  ////// Running Experiment
  auto k = korali::Engine();

  k["Conduit"]["Type"] = "Distributed";
  korali::setKoraliMPIComm(MPI_COMM_WORLD);
  k.run(e);
}
