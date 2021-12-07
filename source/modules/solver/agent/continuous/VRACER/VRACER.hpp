/** \namespace continuous
* @brief Namespace declaration for modules of type: continuous.
*/

/** \file
* @brief Header file for module: VRACER.
*/

/** \dir solver/agent/continuous/VRACER
* @brief Contains code, documentation, and scripts for module: VRACER.
*/

#pragma once

#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/problem/reinforcementLearning/continuous/continuous.hpp"
#include "modules/solver/agent/continuous/continuous.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
namespace continuous
{
;

/**
* @brief Class declaration for module: VRACER.
*/
class VRACER : public Continuous
{
  public: 
  /**
  * @brief [Internal Use] Effective Minibatch Sized used in Multi Agent Reinforcement Learning.
  */
   size_t _effectiveMinibatchSize;
  
 
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
  std::vector<learner::DeepSupervisor *> _criticPolicyLearner;

  /**
   * @brief Korali experiment for obtaining the agent's action
   */
  std::vector<korali::Experiment> _criticPolicyExperiment;

  /**
   * @brief Pointer to actor's experiment problem
   */
  std::vector<problem::SupervisedLearning *> _criticPolicyProblem;

  /**
   * @brief Update the V-target or current and previous experiences in the episode
   * @param expId Current Experience Id
   */
  void updateVtbc(size_t expId);

  /**
   * @brief Calculates the gradients for the policy/critic neural network
   * @param miniBatch The indexes of the experience mini batch
   */
  void calculatePolicyGradients(const std::vector<std::pair<size_t,size_t>> &miniBatch, const std::vector<policy_t> &policy, const size_t policyIdx);

  void runPolicy(const std::vector<std::vector<std::vector<float>>> &stateBatch, std::vector<policy_t> &policy, const size_t policyIdx = 0) override;

  /**
   * @brief [Statistics] Keeps track of the max policy mu of the current minibatch for each action variable
   */
  std::vector<float> _miniBatchPolicyMean;

  /**
   * @brief [Statistics] Keeps track of the max policy sigma of the current minibatch for each action variable
   */
  std::vector<float> _miniBatchPolicyStdDev;

  knlohmann::json getAgentPolicy() override;
  void setAgentPolicy(const knlohmann::json &hyperparameters) override;
  void trainPolicy() override;
  void printAgentInformation() override;
  void initializeAgent() override;
};

} //continuous
} //agent
} //solver
} //korali
;
