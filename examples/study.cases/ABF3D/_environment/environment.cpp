//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "environment.hpp"
#include <sys/stat.h>
#include <unistd.h>

std::string _resultDir;
bool _isTraining;
std::mt19937 _randomGenerator;
std::unique_ptr<msode::rl::MSodeEnvironment> _environment;

void runEnvironment(korali::Sample &s)
{
  // Changing to results directory
  chdir(_resultDir.c_str());

  // Setting seed
  size_t sampleId = s["Sample Id"];

  // Reseting environment and setting initial conditions
  _environment->reset(_randomGenerator, sampleId, true);

  // Setting initial state
  auto state = _environment->getState();

  s["State"] = state;
  state[0] = state[0] / 20.0f; // Swimmer 1 - Pos X
  state[1] = state[1] / 20.0f; // Swimmer 1 - Pos Y
  state[2] = state[2] / 20.0f; // Swimmer 1 - Pos Z
  state[7] = state[7] / 20.0f; // Swimmer 2 - Pos X
  state[8] = state[8] / 20.0f; // Swimmer 2 - Pos Y
  state[9] = state[9] / 20.0f; // Swimmer 2 - Pos Z

  // Defining status variable that tells us whether when the simulation is done
  Status status{Status::Running};

  // Storing action index
  size_t curActionIndex = 0;

  // Starting main environment loop
  while (status == Status::Running)
  {
    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Scaling to lower-upper bounds
    auto [lowerBounds, upperBounds] = _environment->getActionBounds();
    for (size_t i = 0; i < action.size(); i++)
    {
        if (action[i] > upperBounds[i]) action[i] = upperBounds[i];
        else if (action[i] < lowerBounds[i]) action[i] = lowerBounds[i];
        action[i] = action[i] * (upperBounds[i] - lowerBounds[i]) + lowerBounds[0];
    }

    //action[0] = action[0] * (upperBounds[0] - lowerBounds[0]) + lowerBounds[0];

    // Printing Action:
    // if (curActionIndex % 20 == 0)
    //{
    //  printf("Action %lu: [ %f", curActionIndex, action[0]);
    //  for (size_t i = 1; i < action.size(); i++) printf(", %f", action[i]);
    //  printf("]\n");
    //}

    // Setting action
    status = _environment->advance(action);

    // Storing reward
    s["Reward"] = _environment->getReward();

    // Storing new state
    auto state = _environment->getState();

    state[0] = state[0] / 20.0f; // Swimmer 1 - Pos X
    state[1] = state[1] / 20.0f; // Swimmer 1 - Pos Y
    state[2] = state[2] / 20.0f; // Swimmer 1 - Pos Z
    state[7] = state[7] / 20.0f; // Swimmer 2 - Pos X
    state[8] = state[8] / 20.0f; // Swimmer 2 - Pos Y
    state[9] = state[9] / 20.0f; // Swimmer 2 - Pos Z
    s["State"] = state;

    //  Printing State:
    //    if (curActionIndex % 100 == 0)
    //    {
    //      printf("State %lu: [ %.3f", curActionIndex, state[0]);
    //      for (size_t i = 1; i < state.size(); i++) printf(", %.3f", state[i]);
    //      printf("]\n");
    //    }

    // Increasing action count
    curActionIndex++;
  }

  // Setting finalization status
  if (status == Status::Success)
    s["Termination"] = "Terminal";
  else
    s["Termination"] = "Truncated";

  chdir("..");
}

void initializeEnvironment(const std::string confFileName)
{
  std::ifstream confFile(confFileName);

  if (!confFile.is_open())
  {
    fprintf(stderr, "Could not open the config file '%s'", confFileName.c_str());
    exit(-1);
  }

  const Config config = json::parse(confFile);
  _environment = rl::factory::createEnvironment(config, ConfPointer(""));

  // Creating result directory
  mkdir(_resultDir.c_str(), S_IRWXU);
}
