//  Korali environment for CubismUP-3D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "swimmerEnvironment.hpp"

#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

// Swimmer following an obstacle
void runEnvironment(korali::Sample &s)
{
  // Get MPI communicator
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();

  // Get rank and size of subcommunicator
  int rank, size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  // Get rank in world
  int rankGlobal;
  MPI_Comm_rank(MPI_COMM_WORLD,&rankGlobal);

  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%03d", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), rankGlobal/size);
  if( rank == 0 )
  if( not std::filesystem::exists(resDir) )
  if( not std::filesystem::create_directories(resDir) )
  {
    fprintf(stderr, "[Korali] Error creating results directory for environment: %s.\n", resDir);
    exit(-1);
  };

  // Redirecting all output to the log file
  FILE * logFile;
  if( rank == 0 ) {
    char logFilePath[128];
    sprintf(logFilePath, "%s/log.txt", resDir);
    logFile = freopen(logFilePath, "a", stdout);
    if (logFile == NULL)
    {
      printf("[Korali] Error creating log file: %s.\n", logFilePath);
      exit(-1);
    }
  }
  // Make sure folder / logfile is created before switching path
  MPI_Barrier(comm);

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  if( rank==0 ) {
    std::cout << "=======================================================================\n";
    std::cout << "Cubism UP 3D (velocity-pressure 3D incompressible Navier-Stokes solver)\n";
    std::cout << "=======================================================================\n";
    #ifdef NDEBUG
    std::cout << "Running in RELEASE mode!\n";
    #else
    std::cout << "Running in DEBUG mode!\n";
    #endif
  }

  // Creating simulation environment
  ArgumentParser parser(_argc, _argv);
  Simulation *_environment = new Simulation(comm, parser);

  //////////// Obtaining agents (all of them!) ////////////
  // std::vector<std::shared_ptr<Obstacle>> shapes = _environment->getObstacleVector();
  // size_t nAgents = shapes.size();
  // std::vector<StefanFish *> agents(nAgents);
  // for( size_t i = 0; i<nAgents; i++ )
  //   agents[i] = dynamic_cast<StefanFish *>(shapes[i].get());
  /////////////////////////////////////////////////////////

  //////////// Obtaining agents (only followers!) ////////////
  std::vector<std::shared_ptr<Obstacle>> shapes = _environment->getObstacleVector();
  size_t nAgents = shapes.size()-5;
  std::vector<StefanFish *> agents(nAgents);
  // plane 1 with 4 fish
  agents[0] = dynamic_cast<StefanFish *>(shapes[2].get());
  agents[1] = dynamic_cast<StefanFish *>(shapes[3].get());
  agents[2] = dynamic_cast<StefanFish *>(shapes[4].get());
  // # plane 2 with 9 fish
  agents[3]  = dynamic_cast<StefanFish *>(shapes[6].get());
  agents[4]  = dynamic_cast<StefanFish *>(shapes[7].get());
  agents[5]  = dynamic_cast<StefanFish *>(shapes[8].get());
  agents[6]  = dynamic_cast<StefanFish *>(shapes[9].get());
  agents[7]  = dynamic_cast<StefanFish *>(shapes[10].get());
  agents[8]  = dynamic_cast<StefanFish *>(shapes[11].get());
  agents[9]  = dynamic_cast<StefanFish *>(shapes[12].get());
  agents[10] = dynamic_cast<StefanFish *>(shapes[13].get());
  // plane 3 with 4 fish
  agents[11] = dynamic_cast<StefanFish *>(shapes[15].get());
  agents[12] = dynamic_cast<StefanFish *>(shapes[16].get());
  agents[13] = dynamic_cast<StefanFish *>(shapes[17].get());
  ////////////////////////////////////////////////////////////

  // Establishing environment's dump frequency
  _environment->sim.saveTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Setting random initial conditions
  for( size_t i = 0; i<nAgents; i++ )
    setInitialConditions(agents[i], i, s["Mode"] == "Training", rank);
  // After moving the agent, the grid has to be refined
  _environment->refineGrid();

  // Setting initial state
  if( rank == 0 ) {
    if( nAgents > 1 )
    {
      std::vector<std::vector<double>> states(nAgents);
      for( size_t i = 0; i<nAgents; i++ )
      {
        std::vector<double> state = agents[i]->state();
        // assign state/reward to container
        states[i]  = state;
      }
      s["State"] = states;
    }
    else
      s["State"] = agents[0]->state();
  }

  // Variables for time and step conditions
  double t = 0;        // Current time
  size_t curStep = 0;  // current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action     

  // Setting maximum number of steps before truncation
  size_t maxSteps = 200;

  // File to write actions (TODO)
  // std::stringstream filename;
  // filename<<"actions.txt";
  // ofstream myfile(filename.str().c_str());

  // Careful, hardcoded the number of action(s)!
  std::vector<std::vector<double>> actions(nAgents, std::vector<double>(2));

  // Starting main environment loop
  bool done = false;
  while ( curStep < maxSteps && done == false )
  {
    if( rank == 0 ) {
      // Getting new action(s)
      s.update();
      auto actionsJson = s["Action"];

      // Setting action for each agent
      for( size_t i = 0; i<nAgents; i++ )
      {
        std::vector<double> action;
        if( nAgents > 1 )
          action = actionsJson[i].get<std::vector<double>>();
        else
          action = actionsJson.get<std::vector<double>>();
        actions[i] = action;
      }
    }

    // Broadcast and apply action(s) [Careful, hardcoded the number of action(s)!]
    for( size_t i = 0; i<nAgents; i++ )
    {
      MPI_Bcast( actions[i].data(), 2, MPI_DOUBLE, 0, comm );
      if( actions[i].size() != 2 ) std::cout << "Korali returned the wrong number of actions " << actions[i].size() << "\n";
      agents[i]->act(t, actions[i]);
    }

    // Run the simulation until next action is required
    dtAct = 0.;
    for( size_t i = 0; i<nAgents; i++ )
    if( dtAct < agents[i]->getLearnTPeriod() * 0.5 )
      dtAct = agents[i]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;       
    while ( t < tNextAct && done == false )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->timestep(dt);

      // Check if there was a collision -> termination.
      done = _environment->sim.bCollision;

      // Check termination because leaving margins
      for( size_t i = 0; i<nAgents; i++ )
        done = ( done || isTerminal( agents[i] ) );
    }

    // Get and store state and action
    if( rank == 0 ) {
      if( nAgents > 1 )
      {
        std::vector<std::vector<double>> states(nAgents);
        std::vector<double> rewards(nAgents);
        for( size_t i = 0; i<nAgents; i++ )
        {
          std::vector<double> state = agents[i]->state();
          // assign state/reward to container
          states[i]  = state;
          rewards[i] = done ? -10.0 : agents[i]->EffPDefBnd;
        }
        s["State"]  = states;
        s["Reward"] = rewards;
      }
      else{
        s["State"]  = agents[0]->state();
        s["Reward"] = done ? -10.0 : agents[0]->EffPDefBnd;
      }

      // Printing Information:
      printf("[Korali] -------------------------------------------------------\n");
      printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
      if( nAgents > 1 )
      {
        for( size_t i = 0; i<nAgents; i++ )
        {
          auto state  = s["State"][i].get<std::vector<float>>();
          auto action = s["Action"][i].get<std::vector<float>>();
          auto reward = s["Reward"][i].get<float>();
          printf("[Korali] AGENT %ld/%ld\n", i, nAgents);
          printf("[Korali] State: [ %.3f", state[0]);
          for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
          printf("]\n");
          printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
          printf("[Korali] Reward: %.3f\n", reward);
          printf("[Korali] Terminal?: %d\n", done);
          printf("[Korali] -------------------------------------------------------\n");
        }
      }
      else{
        auto state  = s["State"].get<std::vector<float>>();
        auto action = s["Action"].get<std::vector<float>>();
        auto reward = s["Reward"].get<float>();
        printf("[Korali] State: [ %.3f", state[0]);
        for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
        printf("]\n");
        printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
        printf("[Korali] Reward: %.3f\n", reward);
        printf("[Korali] Terminal?: %d\n", done);
        printf("[Korali] -------------------------------------------------------\n");
      }
      fflush(stdout);
    }
    // Advancing to next step
    curStep++;
  }

  // if ( rank == 0 ) {
  //   // Close file to write actions
  //   myfile.close();
  // }

  // Flush CUP logger
  logger.flush();

  // Closing log file
  if( rank == 0 )
    fclose(logFile);

  // delete simulation class
  delete _environment;

  // Setting finalization status
  if( rank == 0 ) {
    if (done == true)
      s["Termination"] = "Terminal";
    else
      s["Termination"] = "Truncated";
  }

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);
}

void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining, int rank)
{
  // Initial fixed conditions
  double initialAngle = 0.0;
  std::array<double,3> initialPosition = agent->getInitialLocation();

  // with noise
  if (isTraining)
  {
    std::uniform_real_distribution<double> disA(-5. / 180. * M_PI, 5. / 180. * M_PI);
    std::uniform_real_distribution<double> disX(-0.025, 0.025);
    std::uniform_real_distribution<double> disY(-0.05, 0.05);
    std::uniform_real_distribution<double> disZ(-0.05, 0.05);

    initialAngle = disA(_randomGenerator);
    initialPosition[0] = initialPosition[0] + disX(_randomGenerator);
    initialPosition[1] = initialPosition[1] + disY(_randomGenerator);
    initialPosition[2] = initialPosition[2] + disZ(_randomGenerator);
  }

  if( rank == 0 )
  {
    printf("[Korali] Initial Condition Agent %ld:\n", agentId);
    printf("[Korali] angle: %f\n", initialAngle);
    printf("[Korali] x: %f\n", initialPosition[0]);
    printf("[Korali] y: %f\n", initialPosition[1]);
    printf("[Korali] z: %f\n", initialPosition[2]);
  }

  // Write initial condition to file
  // std::stringstream filename;
  // filename<<"initialCondition.txt";
  // ofstream myfile(filename.str().c_str(), std::ofstream::app);
  // if (myfile.is_open()) {
  //   myfile << agentId << " " << initialAngle << " " << initialPosition[0] << " " << initialPosition[1] << " " << initialPosition[2] << std::endl;
  //   myfile.close();
  // }
  // else {
  //   fprintf(stderr, "Unable to open %s file...\n", filename.str().c_str());
  //   exit(-1);
  // }

  // Setting initial position and orientation for the fish
  agent->setCenterOfMass(initialPosition);
  agent->setOrientation(initialAngle);
}

bool isTerminal(StefanFish *agent)
{
  double xMin = 0.4;
  double xMax = 2.0;
  
  double yMin = 0.6;
  double yMax = 1.4;

  double zMin = 0.6;
  double zMax = 1.4;

  const double X = agent->position[0];
  const double Y = agent->position[1];
  const double Z = agent->position[2];

  bool terminal = false;
  if (X < xMin) terminal = true;
  if (X > xMax) terminal = true;
  if (Y < yMin) terminal = true;
  if (Y > yMax) terminal = true;
  if (Z < zMin) terminal = true;
  if (Z > zMax) terminal = true;

  return terminal;
}
