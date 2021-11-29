#include "engine.hpp"
#include "modules/solver/agent/continuous/VRACER/VRACER.hpp"
#include "omp.h"
#include "sample/sample.hpp"

#include <gsl/gsl_sf_psi.h>

namespace korali
{
namespace solver
{
namespace agent
{
namespace continuous
{
;

void VRACER::initializeAgent()
{
  // Initializing common discrete agent configuration
  Continuous::initializeAgent();

  // Init statistics
  _statisticsAverageActionSigmas.resize(_problem->_actionVectorSize);

  /*********************************************************************
   * Initializing Critic/Policy Neural Network Optimization Experiment
   *********************************************************************/
  _criticPolicyLearner.resize(_problem->_policiesPerEnvironment);
  _criticPolicyExperiment.resize(_problem->_policiesPerEnvironment);
  _criticPolicyProblem.resize(_problem->_policiesPerEnvironment);

  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
  {
    _criticPolicyExperiment[p]["Problem"]["Type"] = "Supervised Learning";
    _criticPolicyExperiment[p]["Problem"]["Max Timesteps"] = _timeSequenceLength;
    if ( _problem->_policiesPerEnvironment == 1 || _multiPolicyUpdate == "All" )
      _criticPolicyExperiment[p]["Problem"]["Training Batch Size"] = _miniBatchSize * _problem->_policiesPerEnvironment;
    else
      _criticPolicyExperiment[p]["Problem"]["Training Batch Size"] = _miniBatchSize;
    _criticPolicyExperiment[p]["Problem"]["Inference Batch Size"] = 1;
    _criticPolicyExperiment[p]["Problem"]["Input"]["Size"] = _problem->_stateVectorSize;
    _criticPolicyExperiment[p]["Problem"]["Solution"]["Size"] = 1 + _policyParameterCount;

    _criticPolicyExperiment[p]["Solver"]["Type"] = "Learner/DeepSupervisor";
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
    _criticPolicyExperiment[p]["Solver"]["Learning Rate"] = _currentLearningRate;
    _criticPolicyExperiment[p]["Solver"]["Loss Function"] = "Direct Gradient";
    _criticPolicyExperiment[p]["Solver"]["Steps Per Generation"] = 1;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Engine"] = _neuralNetworkEngine;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
    _criticPolicyExperiment[p]["Solver"]["Output Weights Scaling"] = 0.001;

    // No transformations for the state value output
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][0] = "Identity";
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][0] = 1.0f;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][0] = 0.0f;

    // Setting transformations for the selected policy distribution output
    for (size_t i = 0; i < _policyParameterCount; i++)
    {
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][i + 1] = _policyParameterTransformationMasks[i];
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][i + 1] = _policyParameterScaling[i];
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][i + 1] = _policyParameterShifting[i];
    }

    // Running initialization to verify that the configuration is correct
    _criticPolicyExperiment[p].initialize();
    _criticPolicyProblem[p] = dynamic_cast<problem::SupervisedLearning *>(_criticPolicyExperiment[p]._problem);
    _criticPolicyLearner[p] = dynamic_cast<solver::learner::DeepSupervisor *>(_criticPolicyExperiment[p]._solver);
  }

  _maxMiniBatchPolicyMean.resize(_problem->_agentsPerEnvironment);
  _maxMiniBatchPolicyStdDev.resize(_problem->_agentsPerEnvironment);

  _minMiniBatchPolicyMean.resize(_problem->_agentsPerEnvironment);
  _minMiniBatchPolicyStdDev.resize(_problem->_agentsPerEnvironment);

  for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
  {
    _maxMiniBatchPolicyMean[d].resize(_problem->_actionVectorSize);
    _maxMiniBatchPolicyStdDev[d].resize(_problem->_actionVectorSize);
    _minMiniBatchPolicyMean[d].resize(_problem->_actionVectorSize);
    _minMiniBatchPolicyStdDev[d].resize(_problem->_actionVectorSize);
  }
}

void VRACER::trainPolicy()
{
  // Obtaining Minibatch experience indices
  const auto miniBatch = generateMiniBatch(_miniBatchSize);

  /* Gathering state sequences for selected minibatch
   * Shape [ _miniBatchSize * _problem->_agentsPerEnvironment, timeSequenceLength, _problem->_stateVectorSize (+ _problem->_actionVectorSize) ]
   * For _multiAgentSampling == "Single" the distribution with respect to agent will be uniform
   * Otherwise, _miniBatchSize experiences per agent are stacked sequentially [s^{(1)}_1,..., s^{(1)}_M, s^{(2)}_1, ...]
  */ 
  const auto stateSequence = getMiniBatchStateSequence(miniBatch);

  /* Forward Policy, compute Gradient, and perform Backpropagation */

  // Use experiences from all agents to update all policies
  if( _multiPolicyUpdate == "All" )
  {
    for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    {
      // Running policy NN on the Minibatch experiences
      std::vector<policy_t> policyInfo;

      // Forward Mini Batch with all Experiences
      runPolicy(stateSequence, policyInfo, p);

      // Update Metadata and everything needed for gradient computation
      updateExperienceMetadata(miniBatch, policyInfo, p);

      // Now calculating policy gradients
      calculatePolicyGradients(miniBatch, policyInfo, p);

      // Updating learning rate for critic/policy learner guided by REFER
      _criticPolicyLearner[p]->_learningRate = _currentLearningRate;

      // Now applying gradients to update policy NN
      _criticPolicyLearner[p]->runGeneration();
    }
  }

  // Correct Metadata and perform update only using agent's experiences 
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
  {
    // Running policy NN on the Minibatch experiences
    std::vector<policy_t> policyInfo;

    // Forward Mini Batch with all Experiences
    const std::vector<std::vector<std::vector<float>>> stateSequenceTrucated(stateSequence.begin()+p*_problem->_agentsPerEnvironment,stateSequence.begin()+(p+1)*_problem->_agentsPerEnvironment);
    runPolicy(stateSequenceTrucated, policyInfo, p);

    // Update Metadata and everything needed for gradient computation
    updateExperienceMetadata(miniBatch, policyInfo, p);

    // If wished, update the policy only using the experiences from the respective agents
    if( _multiPolicyUpdate == "Self" )
    {
      // Now calculating policy gradients
      calculatePolicyGradients(miniBatch, policyInfo, p);

      // Updating learning rate for critic/policy learner guided by REFER
      _criticPolicyLearner[p]->_learningRate = _currentLearningRate;

      // Now applying gradients to update policy NN
      _criticPolicyLearner[p]->runGeneration();
    }
  }
}

void VRACER::calculatePolicyGradients(const std::vector<size_t> &miniBatch, const std::vector<policy_t> &policyData, const size_t policyIdx)
{
  // Resetting statistics
  std::fill(_statisticsAverageActionSigmas.begin(), _statisticsAverageActionSigmas.end(), 0.0);

  const size_t miniBatchSize = miniBatch.size();
  const size_t policyDataSize = policyData.size();

  for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    _maxMiniBatchPolicyMean[d][i] = -Inf;
    _maxMiniBatchPolicyStdDev[d][i] = -Inf;
    _minMiniBatchPolicyMean[d][i] = +Inf;
    _minMiniBatchPolicyStdDev[d][i] = +Inf;
  }

#pragma omp parallel for
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current experiment
    size_t expId = miniBatch[b];

    // Get state, action and policy for this experience
    const auto &expPolicy = _expPolicyVector[expId];
    const auto &expAction = _actionVector[expId];

    // Gathering metadata
    const auto &curPolicy = _curPolicyVector[expId];
    const auto &expVtbc = _retraceValueVector[expId];

    // .. if cooporative setting average value function
    std::vector<float> expV = _stateValueVector[expId];
    if (_multiAgentRelationship == "Cooperation")
    {
      float avgV = std::accumulate(expV.begin(), expV.end(), 0.);
      avgV /= _problem->_agentsPerEnvironment;
      expV = std::vector<float>(_problem->_agentsPerEnvironment, avgV);
    }

    // Calculate product of importance weights
    float logProdImportanceWeight = 0.0f;
    if (_multiAgentCorrelation)
    {
      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
        logProdImportanceWeight += std::log(_importanceWeightVector[expId][d]);
    }
    float prodImportanceWeight = std::exp(logProdImportanceWeight);

    // No Inner Loop for Self
    size_t numOtherExperiences = _problem->_agentsPerEnvironment;
    if( miniBatchSize == policyDataSize )
      numOtherExperiences = 1;

    for (size_t d = 0; d < numOtherExperiences; d++)
    {
      // Get Agent Id for Experience
      size_t agentExpId = d;
      if( _multiAgentSampling == "Single" )
        agentExpId = _sampledAgentIds[b + d * _problem->_agentsPerEnvironment];
      if( miniBatchSize == policyDataSize )
        agentExpId = policyIdx;

      // Storage for the update gradient
      std::vector<float> gradientLoss(1 + 2 * _problem->_actionVectorSize, 0.0f);

      // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
      gradientLoss[0] = (expVtbc[agentExpId] - expV[agentExpId]);

      // Division from inner gradient
      if (_multiAgentRelationship == "Cooperation")
        gradientLoss[0] /= _problem->_agentsPerEnvironment;

      // Compute policy gradient only if inside trust region (or offPolicy disabled)
      if (_isOnPolicyVector[expId][agentExpId])
      {
        // Qret for terminal state is just reward
        float Qret = getScaledReward(_rewardVector[expId][agentExpId], agentExpId);

        // If experience is non-terminal, add Vtbc
        if (_terminationVector[expId] == e_nonTerminal)
        {
          float nextExpVtbc = _retraceValueVector[expId + 1][agentExpId];
          Qret += _discountFactor * nextExpVtbc;
        }

        // If experience is truncated, add truncated state value
        if (_terminationVector[expId] == e_truncated)
        {
          float nextExpVtbc = _truncatedStateValueVector[expId][agentExpId];
          Qret += _discountFactor * nextExpVtbc;
        }

        // Compute Off-Policy Objective (eq. 5)
        const float lossOffPolicy = Qret - expV[agentExpId];

        // Compute Off-Policy Gradient
        auto polGrad = calculateImportanceWeightGradient(expAction[agentExpId], curPolicy[agentExpId], expPolicy[agentExpId]);

        // If multi-agent correlation, multiply with additional factor
        if (_multiAgentCorrelation)
        {
          const float correlationFactor = prodImportanceWeight / _importanceWeightVector[expId][agentExpId];
          for (size_t i = 0; i < polGrad.size(); i++)
            polGrad[i] *= correlationFactor;
        }

        // Set Gradient of Loss wrt Params
        for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
          gradientLoss[1 + i] = _experienceReplayOffPolicyREFERCurrentBeta[agentExpId] * lossOffPolicy * polGrad[i];
      }

      // Compute derivative of kullback-leibler divergence wrt current distribution params
      const auto klGrad = calculateKLDivergenceGradient(expPolicy[agentExpId], curPolicy[agentExpId]);

      // Step towards old policy (gradient pointing to larger difference between old and current policy)
      const float klGradMultiplier = -(1.0f - _experienceReplayOffPolicyREFERCurrentBeta[agentExpId]);

      // Write vector for backward operation of neural network
      for (size_t i = 0; i < _problem->_actionVectorSize; i++)
      {
        gradientLoss[1 + i] += klGradMultiplier * klGrad[i];
        gradientLoss[1 + i + _problem->_actionVectorSize] += klGradMultiplier * klGrad[i + _problem->_actionVectorSize];

        if (expPolicy[d].distributionParameters[i] > _maxMiniBatchPolicyMean[d][i]) _maxMiniBatchPolicyMean[d][i] = expPolicy[d].distributionParameters[i];
        if (expPolicy[d].distributionParameters[_problem->_actionVectorSize + i] > _maxMiniBatchPolicyStdDev[d][i]) _maxMiniBatchPolicyStdDev[d][i] = expPolicy[d].distributionParameters[_problem->_actionVectorSize + i];
        if (expPolicy[d].distributionParameters[i] < _minMiniBatchPolicyMean[d][i]) _minMiniBatchPolicyMean[d][i] = expPolicy[d].distributionParameters[i];
        if (expPolicy[d].distributionParameters[_problem->_actionVectorSize + i] < _minMiniBatchPolicyStdDev[d][i]) _minMiniBatchPolicyStdDev[d][i] = expPolicy[d].distributionParameters[_problem->_actionVectorSize + i];

        if (std::isfinite(gradientLoss[i + 1]) == false)
        {
          // if ((_multiAgentRelationship == "Cooperation") && (_multiAgentCorrelation == false))
          //   gradientLoss[i + 1] = 0.0;
          // else
          KORALI_LOG_ERROR("Gradient loss for mean returned an invalid value: %f\n", gradientLoss[i + 1]);
        }
        if (std::isfinite(gradientLoss[i + 1 + _problem->_actionVectorSize]) == false)
        {
          // if ((_multiAgentRelationship == "Cooperation") && (_multiAgentCorrelation == false))
          //   gradientLoss[i + 1 + _problem->_actionVectorSize] = 0.0;
          // else
          KORALI_LOG_ERROR("Gradient loss for standard deviation returned an invalid value: %f\n", gradientLoss[i + 1 + _problem->_actionVectorSize]);
        }
      }

      // Set Gradient of Loss as Solution
      _criticPolicyProblem[policyIdx]->_solutionData[b + d * _problem->_agentsPerEnvironment] = gradientLoss;
    }
  }

  // Compute average action stadard deviation
  for (size_t j = 0; j < _problem->_actionVectorSize; j++) _statisticsAverageActionSigmas[j] /= (float)miniBatchSize;
}

float VRACER::calculateStateValue(const std::vector<float> &state, const size_t policyIdx)
{
  // Forward the neural network for this state to get the state value
  const auto evaluation = _criticPolicyLearner[policyIdx]->getEvaluation({std::vector<std::vector<float>>{state}});
  return evaluation[0][0];
}

void VRACER::runPolicy(const std::vector<std::vector<std::vector<float>>> &stateBatch, std::vector<policy_t> &policyInfo, const size_t policyIdx)
{
  // Getting batch size
  size_t batchSize = stateBatch.size();

  // Forward the neural network for this states
  const auto evaluation = _criticPolicyLearner[policyIdx]->getEvaluation(stateBatch);

  // Preparing storage for results
  policyInfo.resize(batchSize);

  // Write results to policy-vector
  for( size_t b = 0; b<batchSize; b++ )
  {
    policyInfo[b].stateValue = evaluation[b][0];
    policyInfo[b].distributionParameters.assign(evaluation[b].begin() + 1, evaluation[b].end());
  }
}

knlohmann::json VRACER::getAgentPolicy()
{
  knlohmann::json hyperparameters;
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    hyperparameters["Policy Hyperparameters"][p] = _criticPolicyLearner[p]->getHyperparameters();
  return hyperparameters;
}

void VRACER::setAgentPolicy(const knlohmann::json &hyperparameters)
{
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    _criticPolicyLearner[p]->setHyperparameters(hyperparameters[p].get<std::vector<float>>());
}

void VRACER::printAgentInformation()
{
  _k->_logger->logInfo("Normal", " + [VRACER] Policy Learning Rate: %.3e\n", _currentLearningRate);
  _k->_logger->logInfo("Detailed", " + [VRACER] Max Policy Parameters (Mu & Sigma):\n");
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    _k->_logger->logInfo("Detailed", " + [VRACER] Action %zu: (%.3e,%.3e)\n", i, _maxMiniBatchPolicyMean[0][i], _maxMiniBatchPolicyStdDev[0][i]);
  _k->_logger->logInfo("Detailed", " + [VRACER] Min Policy Parameters (Mu & Sigma):\n");
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    _k->_logger->logInfo("Detailed", " + [VRACER] Action %zu: (%.3e,%.3e)\n", i, _minMiniBatchPolicyMean[0][i], _minMiniBatchPolicyStdDev[0][i]);
}

void VRACER::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Statistics", "Average Action Sigmas"))
 {
 try { _statisticsAverageActionSigmas = js["Statistics"]["Average Action Sigmas"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ VRACER ] \n + Key:    ['Statistics']['Average Action Sigmas']\n%s", e.what()); } 
   eraseValue(js, "Statistics", "Average Action Sigmas");
 }

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Exploration Noise"))
 {
 try { _k->_variables[i]->_initialExplorationNoise = _k->_js["Variables"][i]["Initial Exploration Noise"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ VRACER ] \n + Key:    ['Initial Exploration Noise']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Exploration Noise");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Exploration Noise'] required by VRACER.\n"); 

 } 
 Continuous::setConfiguration(js);
 _type = "agent/continuous/VRACER";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: VRACER: \n%s\n", js.dump(2).c_str());
} 

void VRACER::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Statistics"]["Average Action Sigmas"] = _statisticsAverageActionSigmas;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Initial Exploration Noise"] = _k->_variables[i]->_initialExplorationNoise;
 } 
 Continuous::getConfiguration(js);
} 

void VRACER::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Continuous::applyModuleDefaults(js);
} 

void VRACER::applyVariableDefaults() 
{

 std::string defaultString = "{\"Initial Exploration Noise\": -1.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Continuous::applyVariableDefaults();
} 

bool VRACER::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Continuous::checkTermination();
 return hasFinished;
}

;

} //continuous
} //agent
} //solver
} //korali
;
