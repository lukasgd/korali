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

// Declare reduction clause for vectors
#pragma omp declare reduction(vec_float_plus : std::vector<float> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus <float>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

void VRACER::initializeAgent()
{
  // Initializing common discrete agent configuration
  Continuous::initializeAgent();

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
    _criticPolicyExperiment[p]["Problem"]["Training Batch Size"] = _miniBatchSize;
    if( _multiPolicyUpdate == "Together" )
      _criticPolicyExperiment[p]["Problem"]["Training Batch Size"] = _miniBatchSize * _problem->_agentsPerEnvironment;
    _criticPolicyExperiment[p]["Problem"]["Testing Batch Size"] = 1;
    _criticPolicyExperiment[p]["Problem"]["Input"]["Size"] = _problem->_stateVectorSize;
    _criticPolicyExperiment[p]["Problem"]["Solution"]["Size"] = 1 + _policyParameterCount;

    _criticPolicyExperiment[p]["Solver"]["Type"] = "Learner/DeepSupervisor";
    _criticPolicyExperiment[p]["Solver"]["Mode"] = "Training";
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
    _criticPolicyExperiment[p]["Solver"]["Learning Rate"] = _currentLearningRate;
    _criticPolicyExperiment[p]["Solver"]["Loss Function"] = "Direct Gradient";
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

  _miniBatchPolicyMean.resize(_problem->_actionVectorSize);
  _miniBatchPolicyStdDev.resize(_problem->_actionVectorSize);
}

void VRACER::trainPolicy()
{
  // Obtaining minibatch of experiences
  auto miniBatch = generateMiniBatch();

  // Gathering state sequences for selected minibatch
  const auto stateSequence = getMiniBatchStateSequence(miniBatch);

  /* Forward Policy, compute Gradient, and perform Backpropagation */
  // Update using a large minibatch
  if( _multiPolicyUpdate == "Together" )
  {
    // Running policy NN on the Minibatch experiences
    std::vector<policy_t> policyInfo;

    // Forward Mini Batch
    runPolicy(stateSequence, policyInfo, 0);

    // Update Metadata and everything needed for gradient computation
    updateExperienceMetadata(miniBatch, policyInfo, 0);

    // Now calculating policy gradients
    calculatePolicyGradients(miniBatch, policyInfo, 0);

    // Updating learning rate for critic/policy learner guided by REFER
    _criticPolicyLearner[0]->_learningRate = _currentLearningRate;

    // Now applying gradients to update policy NN
    _criticPolicyLearner[0]->runGeneration();
  }
  else
  {
    for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
    {
      // Skip iterations depending on case
      if( _multiPolicyUpdate == "All" && p == a )
        continue;

      if( _multiPolicyUpdate == "Own" && p != a )
        continue;

      // Running policy NN on the Minibatch experiences
      std::vector<policy_t> policyInfo;

      // Extract minibatch of experiences for agent a
      std::vector<std::pair<size_t,size_t>> miniBatchTruncated( miniBatch.begin()+a*_miniBatchSize, miniBatch.begin()+(a+1)*_miniBatchSize );
      const std::vector<std::vector<std::vector<float>>> stateSequenceTruncated( stateSequence.begin()+a*_miniBatchSize, stateSequence.begin()+(a+1)*_miniBatchSize );

      // Forward Mini Batch
      runPolicy(stateSequenceTruncated, policyInfo, p);

      // Update Metadata and everything needed for gradient computation
      updateExperienceMetadata(miniBatchTruncated, policyInfo, p);

      // Now calculating policy gradients
      calculatePolicyGradients(miniBatchTruncated, policyInfo, p);

      // Updating learning rate for critic/policy learner guided by REFER
      _criticPolicyLearner[p]->_learningRate = _currentLearningRate;

      // Now applying gradients to update policy NN
      _criticPolicyLearner[p]->runGeneration();
    }

    // Now updated with own experiences
    if( _multiPolicyUpdate == "All" )
    {
      for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
      {
        // Running policy NN on the Minibatch experiences
        std::vector<policy_t> policyInfo;

        // Extract minibatch of experiences for agent p
        std::vector<std::pair<size_t,size_t>> miniBatchTruncated( miniBatch.begin()+p*_miniBatchSize, miniBatch.begin()+(p+1)*_miniBatchSize );
        const std::vector<std::vector<std::vector<float>>> stateSequenceTruncated( stateSequence.begin()+p*_miniBatchSize, stateSequence.begin()+(p+1)*_miniBatchSize );

        // Forward Mini Batch
        runPolicy(stateSequenceTruncated, policyInfo, p);

        // Update Metadata and everything needed for gradient computation
        updateExperienceMetadata(miniBatchTruncated, policyInfo, p);

        // Now calculating policy gradients
        calculatePolicyGradients(miniBatchTruncated, policyInfo, p);

        // Updating learning rate for critic/policy learner guided by REFER
        _criticPolicyLearner[p]->_learningRate = _currentLearningRate;

        // Now applying gradients to update policy NN
        _criticPolicyLearner[p]->runGeneration();
      }
    }
  }
}

void VRACER::calculatePolicyGradients(const std::vector<std::pair<size_t,size_t>>  &miniBatch, const std::vector<policy_t> &policyData, const size_t policyIdx)
{
  const size_t miniBatchSize = miniBatch.size();

  #pragma omp parallel for reduction(vec_float_plus: _miniBatchPolicyMean, _miniBatchPolicyStdDev)
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current experiment
    const size_t expId   = miniBatch[b].first;
    const size_t agentId = miniBatch[b].second;

    // Get data for this experience
    const auto &expPolicy = _expPolicyVector[expId][agentId];
    const auto &expAction = _actionVector[expId][agentId];
    const auto &expReward = _rewardVector[expId][agentId];
    const auto &expVtbc = _retraceValueVector[expId][agentId];
    const auto &importanceWeights = _importanceWeightVector[expId];
    const auto &isOnPolicy = _isOnPolicyVector[expId][agentId];
    const auto &curPolicy = _curPolicyVector[expId][agentId];

    std::vector<float> expValues = _stateValueVector[expId];

    // For cooporative setting average value function
    if (_multiAgentRelationship == "Cooperation")
    {
      float avgV = std::accumulate(expValues.begin(), expValues.end(), 0.);
      avgV /= _problem->_agentsPerEnvironment;
      expValues = std::vector<float>(_problem->_agentsPerEnvironment, avgV);
    }

    const auto &expV = expValues[agentId];

    // Storage for the update gradient
    std::vector<float> gradientLoss(1 + 2 * _problem->_actionVectorSize, 0.0f);

    // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
    gradientLoss[0] = (expVtbc - expV);

    // Division from inner gradient
    if (_multiAgentRelationship == "Cooperation")
      gradientLoss[0] /= _problem->_agentsPerEnvironment;

    // Check gradient
    if (std::isfinite(gradientLoss[0]) == false)
      KORALI_LOG_ERROR("Gradient loss for value returned an invalid value: %f\n", gradientLoss[0]);

    // Compute policy gradient only if inside trust region (or offPolicy disabled)
    if ( isOnPolicy )
    {
      // Qret for terminal state is just reward
      float Qret = getScaledReward(expReward, agentId);

      // If experience is non-terminal, add Vtbc
      if (_terminationVector[expId] == e_nonTerminal)
      {
        float nextExpVtbc = _retraceValueVector[expId + 1][agentId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // If experience is truncated, add truncated state value
      if (_terminationVector[expId] == e_truncated)
      {
        float nextExpVtbc = _truncatedStateValueVector[expId][agentId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // Compute Off-Policy Objective (eq. 5)
      const float lossOffPolicy = Qret - expV;

      // Compute Off-Policy Gradient
      auto polGrad = calculateImportanceWeightGradient(expAction, curPolicy, expPolicy);

      // If multi-agent correlation, multiply with additional factor
      if (_multiAgentCorrelation)
      {
        // Calculate product of importance weights
        float logProdImportanceWeight = 0.0f;
        for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
          logProdImportanceWeight += std::log(importanceWeights[d]);
        float prodImportanceWeight = std::exp(logProdImportanceWeight);

        const float correlationFactor = prodImportanceWeight / importanceWeights[agentId];
        for (size_t i = 0; i < polGrad.size(); i++)
          polGrad[i] *= correlationFactor;
      }

      // Set Gradient of Loss wrt Params
      for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
        gradientLoss[1 + i] = _experienceReplayOffPolicyREFERCurrentBeta[agentId] * lossOffPolicy * polGrad[i];
    }

    // Compute derivative of kullback-leibler divergence wrt current distribution params
    const auto klGrad = calculateKLDivergenceGradient(expPolicy, curPolicy);

    // Step towards old policy (gradient pointing to larger difference between old and current policy)
    const float klGradMultiplier = -(1.0f - _experienceReplayOffPolicyREFERCurrentBeta[agentId]);

    // Write vector for backward operation of neural network
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      gradientLoss[1 + i] += klGradMultiplier * klGrad[i];
      gradientLoss[1 + i + _problem->_actionVectorSize] += klGradMultiplier * klGrad[i + _problem->_actionVectorSize];

      if (std::isfinite(gradientLoss[i + 1]) == false)
        KORALI_LOG_ERROR("Gradient loss for mean returned an invalid value: %f\n", gradientLoss[i + 1]);

      if (std::isfinite(gradientLoss[i + 1 + _problem->_actionVectorSize]) == false)
        KORALI_LOG_ERROR("Gradient loss for standard deviation returned an invalid value: %f\n", gradientLoss[i + 1 + _problem->_actionVectorSize]);

      // Update statistics
      _miniBatchPolicyMean[i]   += expPolicy.distributionParameters[i];
      _miniBatchPolicyStdDev[i] += expPolicy.distributionParameters[_problem->_actionVectorSize + i];
    }

    // Set Gradient of Loss as Solution
    _criticPolicyProblem[policyIdx]->_solutionData[b] = gradientLoss;
  }

  // Update Statistics
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  { 
    _miniBatchPolicyMean[i]   /= miniBatchSize;
    _miniBatchPolicyStdDev[i] /= miniBatchSize;
  }

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
  _k->_logger->logInfo("Detailed", " + [VRACER] Average Policy Parameters (Mu & Sigma):\n");
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    _k->_logger->logInfo("Detailed", " + [VRACER] Action %zu: (%.3e,%.3e)\n", i, _miniBatchPolicyMean[i], _miniBatchPolicyStdDev[i]);
}

void VRACER::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

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
