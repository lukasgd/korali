/** \namespace discrete
* @brief Namespace declaration for modules of type: discrete.
*/

/** \file
* @brief Header file for module: dVRACER.
*/

/** \dir solver/agent/discrete/dVRACER
* @brief Contains code, documentation, and scripts for module: dVRACER.
*/

#pragma once

#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/problem/reinforcementLearning/discrete/discrete.hpp"
#include "modules/solver/agent/discrete/discrete.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
namespace discrete
{
;

/**
* @brief Class declaration for module: dVRACER.
*/
class dVRACER : public Discrete
{
  public: 
  /**
  * @brief [Internal Use] Standard deviation of the actions in the minibatch.
  */
   std::vector<float> _statisticsAverageActionSigmas;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  /**
   * @brief Pointer to training the actor network
   */
  learner::DeepSupervisor *_criticPolicyLearner;

  /**
   * @brief Korali experiment for obtaining the agent's action
   */
  korali::Experiment _criticPolicyExperiment;

  /**
   * @brief Pointer to actor's experiment problem
   */
  problem::SupervisedLearning *_criticPolicyProblem;

  /**
   * @brief Update the V-target or current and previous experiences in the episode
   * @param expId Current Experience Id
   */
  void updateVtbc(size_t expId);

  /**
   * @brief Calculates the gradients for the policy/critic neural network
   * @param miniBatch The indexes of the experience mini batch
   */
  void calculatePolicyGradients(const std::vector<size_t> &miniBatch);

  std::vector<policy_t> runPolicy(const std::vector<std::vector<std::vector<float>>> &stateBatch) override;

  knlohmann::json getAgentPolicy() override;
  void setAgentPolicy(const knlohmann::json &hyperparameters) override;
  void trainPolicy() override;
  void printAgentInformation() override;
  void initializeAgent() override;
};

} //discrete
} //agent
} //solver
} //korali
;
