// Select which environment to use
#include "_model/transportEnvironment.hpp"
#include "korali.hpp"

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
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine

  // Setting results path
  std::string trainingResultsPath = "_results_transport_vracer_training/";
  std::string testingResultsPath = "_results_transport_vracer_testing/";

  // Creating Experiment
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  
  // Check if existing results are there and continuing them
  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true) printf("[Korali] Continuing execution from previous run...\n");

  // Configuring Experiment
  e["Problem"]["Environment Function"] = &runEnvironmentVracer;
  e["Problem"]["Training Reward Threshold"] = std::numeric_limits<double>::infinity();
  e["Problem"]["Policy Testing Episodes"] = 5;
  // e["Problem"]["Actions Between Policy Updates"] = 1;

  // Setting results path an dumping frequency in CUP
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;

  // Setting up the state variables (see _deps/CUP-2D/source/obstacles/SmartCylinder.cpp)
  const size_t numStates = 4;
  size_t curVariable = 0;
  for (; curVariable < numStates; curVariable++)
  {
    e["Variables"][curVariable]["Name"] = std::string("StateVar") + std::to_string(curVariable);
    e["Variables"][curVariable]["Type"] = "State";
  }

  const double maxForce = 1e-2;
  const double maxMomentum = maxForce/0.1; // radius == 0.1

  e["Variables"][curVariable]["Name"] = "Force X";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -maxForce;
  e["Variables"][curVariable]["Upper Bound"] = +maxForce;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 0.3*maxForce;

  curVariable++;
  e["Variables"][curVariable]["Name"] = "Force Y";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -maxForce;
  e["Variables"][curVariable]["Upper Bound"] = +maxForce;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 0.3*maxForce;
 
  curVariable++;
  e["Variables"][curVariable]["Name"] = "Momentum";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -maxMomentum;
  e["Variables"][curVariable]["Upper Bound"] = +maxMomentum;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 0.3*maxMomentum;


  /// Defining Agent Configuration
  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Concurrent Environments"] = N;
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Learning Rate"] = 1e-4;
  e["Solver"]["Discount Factor"] = 0.99;
  e["Solver"]["Mini Batch"]["Size"] =  128;

  /// Defining the configuration of replay memory
  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
  e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
  e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
  e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
  e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

  //// Defining Policy distribution and scaling parameters
  e["Solver"]["Policy"]["Distribution"] = "Normal";
  e["Solver"]["State Rescaling"]["Enabled"] = true;
  e["Solver"]["Reward"]["Rescaling"]["Enabled"] = true;

  //// Defining Neural Network
  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  e["Solver"]["Neural Network"]["Optimizer"] = "Adam";

  e["Solver"]["L2 Regularization"]["Enabled"] = true;
  e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32;

  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

  ////// Defining Termination Criteria
  e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e7;
  // e["Solver"]["Termination Criteria"]["Testing"]["Max Generations"] = 1000;
  // e["Solver"]["Termination Criteria"]["Testing"]["Target Average Reward"] = 0.99;

  ////// Setting Korali output configuration
  e["Console Output"]["Verbosity"] = "Detailed";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = trainingResultsPath;

  ////// Running Experiment
  auto k = korali::Engine();

  // Configuring profiler output
  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = trainingResultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 10;

  k["Conduit"]["Type"] = "Distributed";
  korali::setKoraliMPIComm(MPI_COMM_WORLD);

  k.run(e);
}
