#include "windmillEnvironment.hpp"
#include <chrono>
#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

// 4 windmills with variable torque applied to them
void runEnvironment(korali::Sample &s)
{
  ////////////////////////////////////////// setup stuff 
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
  ////////////////////////////////////////// setup stuff 

  ////////////////////////////////////////// Initialize agents and objective
  // Obtaining agent, 4 windmills 
  Windmill* agent1 = dynamic_cast<Windmill*>(_environment->getShapes()[0]);
  Windmill* agent2 = dynamic_cast<Windmill*>(_environment->getShapes()[1]);

  // useful agent functions :
  // void act( double action );
  // double reward( std::array<Real,2> target, std::vector<double> target_vel, double C = 10);
  // std::vector<double> state();

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Setting initial conditions
  setInitialConditions(agent1, 0.0, s["Mode"] == "Training");
  setInitialConditions(agent2, 0.0, s["Mode"] == "Training");
  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Set target
  std::array<Real,2> target_pos{0.69,0.69};
  std::array<Real, 2> target_vel={0.0,0.0};

  agent1->setTarget(target_pos);
  agent2->setTarget(target_pos);

  std::vector<double> state1 = agent1->state();
  std::vector<double> state2 = agent2->state();

  std::vector<double> state = {state1[0], state1[1], state2[0], state2[1]};

  s["State"] = state;

  // Setting initial time and step conditions
  double t = 0;        // Current time
  double tNextAct = 0; // Time until next action
  size_t curStep = 0;  // current Step

  // Setting maximum number of steps before truncation
  size_t maxSteps = 2000; // 2000 for training

  while (curStep < maxSteps)
  {
    // Getting new action
    s.update();

    // Reading new action
    std::vector<double> action = s["Action"];

    // Setting action for 
    agent1->act( action[0] );
    agent2->act( action[1] );

    // Run the simulation until next action is required
    tNextAct += 0.01;
    while ( t < tNextAct )
    {
      // Calculate simulation timestep
      const double dt = std::min(_environment->calcMaxTimestep(), 0.01);
      t += dt;

      // Advance simulation
      _environment->advance(dt);
    }

    // used for "both" run
    // Real en = 5.0e4;
    // Real flow = 2.5;

    // used for "energy_zero" run
    // Real en = 5.0e4;
    // Real flow = 0.0;

    // used for "flow_zero" run
    Real en = 0.0;
    Real flow = 2.5;

    double r1 = agent1->reward( target_vel,  en, flow);
    double r2 = agent2->reward( target_vel,  en, flow);
    double reward = (r1 + r2);

    // Obtaining new agent state
    state1 = agent1->state();
    state2 = agent2->state();
    state = {state1[0], state1[1], state2[0], state2[1]};

    // Printing Information:
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    printf("[Korali] State: [ ");
    for (size_t i = 0; i < state.size(); i++){
      if (i%2 == 0){
        printf("[%.3f, ", state[i]);
      } else {
        printf("%.3f]", state[i]);
      }
    }
    printf("]\n");
    printf("[Korali] Action: [ %.8f, %.8f ]\n", action[0], action[1]);
    printf("[Korali] Reward: %.3f+%.3f=%.3f\n", r1, r2, reward);
    printf("[Korali] -------------------------------------------------------\n");
    fflush(stdout);

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
  s["Termination"] = "Truncated";

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

// set initial conditions of the agent
void setInitialConditions(Windmill* agent, double init_angle, bool randomized)
{
  // Intial fixed condition of angle and angular velocity
  double angle = init_angle;

  // set random beginning angle
  if(randomized)
  {
    // windmills have 3 axis of symmetry, meaning if we rotate it by 120°,we have the same setup
    std::uniform_real_distribution<double> dis(0, 2*(M_PI/3));
    angle = dis(_randomGenerator);
  }
  
  printf("[Korali] Initial Conditions:\n");
  printf("[Korali] orientation: %f\n", angle);

  agent->setOrientation(angle);
}