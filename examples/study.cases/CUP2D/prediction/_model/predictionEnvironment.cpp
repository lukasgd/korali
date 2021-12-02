//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "predictionEnvironment.hpp"
#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

inline std::vector<double> getBlockCenter(ScalarBlock &scalarBlock, BlockInfo blockInfo){

  double p[2];

  int ix = ScalarBlock::sizeX/2;
  int iy = ScalarBlock::sizeY/2;

  blockInfo.pos(p, ix, iy);

  std::vector<double> center(2, 0);
  center[0] = p[0]-0.5*blockInfo.h;
  center[1] = p[1]-0.5*blockInfo.h;

  return center;
}

inline std::vector<double> getBlockAverage(ScalarBlock &scalarBlock, VectorBlock &vectorBlock){
  
  std::vector<double> blockAverage(3, 0);
  for(int iy=0; iy<VectorBlock::sizeY; ++iy)
  for(int ix=0; ix<VectorBlock::sizeX; ++ix)
  {
  // Sum quantity over cells (block specific)
    blockAverage[0] += scalarBlock(ix,iy).s;
    blockAverage[1] += vectorBlock(ix,iy).u[0];
    blockAverage[2] += vectorBlock(ix,iy).u[1];
  }
  for(size_t i = 0; i<blockAverage.size(); ++i)
    blockAverage[i] /= (VectorBlock::sizeX * VectorBlock::sizeY);

  return blockAverage;
};

void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Get rank to create/switch to results folder
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%03u", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), rank);
  if(not std::filesystem::exists(resDir))
  if(not std::filesystem::create_directories(resDir))
  {
    fprintf(stderr, "[Korali] Error creating results directory for environment: %s.\n", resDir);
    exit(-1);
  };

  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "w", stdout);
  if (logFile == NULL)
  {
    printf("[Korali] Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Creating simulation environment
  Simulation *_environment = new Simulation(_argc, _argv);
  _environment->init();

  // Assigning number of agents given number of blocks
  int bpdx = _environment->sim.bpdx;
  int bpdy = _environment->sim.bpdy;
  size_t nAgents = bpdx*bpdy - 2*bpdx - 2*(bpdy - 2);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Defining the state dimension (each agent is a block containing stenciled flow information)
  std::vector<std::vector<double>> states(nAgents, std::vector<double>(15, 0));
  s["State"] = states;

  // Variables for time and step conditions
  size_t curStep = 0;

  // Setting maximum number of steps before truncation
  size_t maxSteps = 1000;

  // Starting main environment loop
  bool done = false;
  double t = 0;
  const double tStart = 30; // set to tAccel (from Cubism) plus some time to allow information propagation (to get to steady-state)
  double tNextAct = tStart + 0;
  const double dtNextAct = 0.1;

  // Advance simulation to desired time before beginning RL loop
  while(t < tStart){

    // Computing CFD timestep
    const double dt = _environment->calcMaxTimestep();

    // Advancing simulation
    _environment->advance(dt);

    // Stepping forward in time
    t += dt;

  }

  while(curStep < maxSteps && done == false){

    // Getting new actions
    s.update();

    // Reading new action(s)
    auto actions = s["Action"];

    // Updating time to the next action
    tNextAct += dtNextAct;

    // Running simulation until next action
    while(t < tNextAct){

    // Computing CFD timestep
    const double dt = _environment->calcMaxTimestep();

    // Advancing simulation
    _environment->advance(dt);

    // Stepping forward in time
    t += dt;

    }

    // Checking termination (not implemented)
    done = (done || isTerminal());

    // Setting reward
    std::vector<double> rewards(nAgents, 0);

    // Getting block information
    const std::vector<BlockInfo>& presInfo = _environment->sim.pres->getBlocksInfo();
    const std::vector<BlockInfo>& velInfo  = _environment->sim.vel->getBlocksInfo();

    // Getting number of blocks
    const int nBlocks = presInfo.size();

    // Getting individual block averages and centers for the entire field
    std::vector<std::vector<double>> fieldAvg(nBlocks, std::vector<double>(3, 0));
    std::vector<std::vector<double>> fieldCenter(nBlocks, std::vector<double>(2, 0));
    for (int b=0; b<nBlocks; ++b){
    
    ScalarBlock & __restrict__ P = *(ScalarBlock*) presInfo[b].ptrBlock;
    VectorBlock & __restrict__ V = *(VectorBlock*)  velInfo[b].ptrBlock;

    auto blockAverage = getBlockAverage(P, V);
    auto blockCenter  = getBlockCenter(P, presInfo[b]);

    fieldAvg[b] = blockAverage;
    fieldCenter[b] = blockCenter;
    }

    // Exploring field and building state vector given candidate stencil
    size_t agentID = 0;
    for (int a = 0; a<nBlocks; a++){
    
      // Setting container for state
      std::vector<double> state;

      // Ignoring blocks at the boundary (as they are NOT agents)
      const BlockInfo & info = velInfo[a];
      if ((info.index[0] == 0)      ||
          (info.index[0] == bpdx-1) ||
          (info.index[1] == 0)      ||
          (info.index[1] == bpdy-1)) 
          continue;
    
      // Getting block information given stencil
      const BlockInfo &infoNeighb_west  = _environment->sim.vel->getBlockInfoAll(velInfo[a].level, info.Znei_(-1, 0, 0));
      const BlockInfo &infoNeighb_east  = _environment->sim.vel->getBlockInfoAll(velInfo[a].level, info.Znei_(1, 0, 0));
      const BlockInfo &infoNeighb_north = _environment->sim.vel->getBlockInfoAll(velInfo[a].level, info.Znei_(0, 1, 0));
      const BlockInfo &infoNeighb_south = _environment->sim.vel->getBlockInfoAll(velInfo[a].level, info.Znei_(0, -1, 0));

      // Obtaining global indices
      int global_ind_west  = infoNeighb_west.blockID;
      int global_ind_east  = infoNeighb_east.blockID;
      int global_ind_north = infoNeighb_north.blockID;
      int global_ind_south = infoNeighb_south.blockID;

      // Extracting averages given stencil (fieldAvg: blocks x 3)
      state.insert(state.end(), fieldAvg[a].begin(), fieldAvg[a].end());
      state.insert(state.end(), fieldAvg[global_ind_west].begin(), fieldAvg[global_ind_west].end());
      state.insert(state.end(), fieldAvg[global_ind_east].begin(), fieldAvg[global_ind_east].end());
      state.insert(state.end(), fieldAvg[global_ind_north].begin(), fieldAvg[global_ind_north].end());
      state.insert(state.end(), fieldAvg[global_ind_south].begin(), fieldAvg[global_ind_south].end());

      // Assigning state to container
      states[agentID] = state;

      // Getting action
      auto action = actions[agentID].get<std::vector<double>>();

      // Computing reward and assigning to container
      for(size_t j = 0; j<3; ++j){
        rewards[agentID] -= (action[j] - state[j]) * (action[j] - state[j]); // / nAgents; // TODO, try different norm & weigths for p,u,v
      }

      // Dump p, u, v. One .csv for each curStep, containing all field data for real/predicted quantities.
      if (s["Mode"] == "Testing" && curStep % 5 == 0){
        
        int numDigits = (int)(std::log10(maxSteps)+1);
        std::string curStepString = std::to_string(curStep);
        std::string fileID = std::string(numDigits - curStepString.length(), '0') + curStepString;

        std::ofstream fields;
        fields.open(fileID + "fields.csv", std::ios_base::app);
        fields << curStep << "," << t << "," << fieldCenter[a][0] << "," << fieldCenter[a][1] << ","
               << fieldAvg[a][0] << "," << fieldAvg[a][1] << "," << fieldAvg[a][2] << ","
               << action[0] << "," << action[1] << "," << action[2] << "\n";
        fields.close();
      }

      agentID++;
    }

    s["State"] = states;
    s["Reward"] = rewards;

    // Advancing to next step
    curStep++;
  }

  // Flushing CUP logger
  logger.flush();

  // Deleting simulation class
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

bool isTerminal()
{
  // IDEA: Compare simulation to prediction and terminate RL is we are too far
  
  return false;
}

// TO DO add randomization, re-run one CFD timestep ahead, add multi-step with own prediction/prediction many steps ahead