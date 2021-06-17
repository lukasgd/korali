#include "engine.hpp"
#include "modules/solver/agent/continuous/continuous.hpp"
#include "sample/sample.hpp"

#include <gsl/gsl_sf_psi.h>

namespace korali
{
namespace solver
{
namespace agent
{


void Continuous::initializeAgent()
{
  // Getting continuous problem pointer
  _problem = dynamic_cast<problem::reinforcementLearning::Continuous *>(_k->_problem);

  // Obtaining action shift and scales for bounded distributions
  _actionShifts.resize(_problem->_actionVectorSize);
  _actionScales.resize(_problem->_actionVectorSize);
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    // For bounded distributions, infinite bounds should result in an error message
    if (_policyDistribution == "Squashed Normal" || _policyDistribution == "Beta")
    {
      if (isfinite(_actionLowerBounds[i]) == false)
        KORALI_LOG_ERROR("Provided lower bound (%f) for action variable %lu is non-finite, but the distribution (%s) is bounded.\n", _actionLowerBounds[i], i, _policyDistribution.c_str());

      if (isfinite(_actionUpperBounds[i]) == false)
        KORALI_LOG_ERROR("Provided upper bound (%f) for action variable %lu is non-finite, but the distribution (%s) is bounded.\n", _actionUpperBounds[i], i, _policyDistribution.c_str());

      _actionShifts[i] = (_actionUpperBounds[i] + _actionLowerBounds[i]) * 0.5f;
      _actionScales[i] = (_actionUpperBounds[i] - _actionLowerBounds[i]) * 0.5f;
    }
  }

  // Obtaining policy parameter transformations (depends on which policy distribution chosen)
  if (_policyDistribution == "Normal" || _policyDistribution == "Squashed Normal")
  {
    _policyParameterCount = 2 * _problem->_actionVectorSize; // Means and Sigmas

    // Allocating space for the required transformations
    _policyParameterTransformationMasks.resize(_policyParameterCount);
    _policyParameterScaling.resize(_policyParameterCount);
    _policyParameterShifting.resize(_policyParameterCount);

    // Establishing transformations for the Normal policy
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      auto varIdx = _problem->_actionVectorIndexes[i];
      float sigma = _k->_variables[varIdx]->_initialExplorationNoise;

      // Checking correct noise configuration
      if (sigma <= 0.0f) KORALI_LOG_ERROR("Provided initial noise (%f) for action variable %lu is not defined or negative.\n", sigma, varIdx);

      // Identity mask for Means
      _policyParameterScaling[i] = 1.0f;
      _policyParameterShifting[i] = 0.0f;
      _policyParameterTransformationMasks[i] = "Identity";

      // Softplus mask for Sigmas
      _policyParameterTransformationMasks[_problem->_actionVectorSize + i] = "Softplus";
      _policyParameterScaling[_problem->_actionVectorSize + i] = sigma * 2.0f;
      _policyParameterShifting[_problem->_actionVectorSize + i] = 0.0f;
    }
  }

  if (_policyDistribution == "Beta")
  {
    _policyParameterCount = 2 * _problem->_actionVectorSize; // Mu and Variance

    // Allocating space for the required transformations
    _policyParameterTransformationMasks.resize(_policyParameterCount);
    _policyParameterScaling.resize(_policyParameterCount);
    _policyParameterShifting.resize(_policyParameterCount);

    // Establishing transformations for the Normal policy
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      auto varIdx = _problem->_actionVectorIndexes[i];
      float variance = _k->_variables[varIdx]->_initialExplorationNoise;

      // Checking correct noise configuration
      if (variance <= 0.0f) KORALI_LOG_ERROR("Provided initial noise (%f) for action variable %lu is not defined or negative.\n", variance, varIdx);

      // Identity mask for Means
      _policyParameterScaling[i] = 1.0f;
      _policyParameterShifting[i] = 0.0f;
      _policyParameterTransformationMasks[i] = "Identity";

      // Sigmoid Mask for Variance
      _policyParameterTransformationMasks[_problem->_actionVectorSize + i] = "Sigmoid";
      _policyParameterScaling[_problem->_actionVectorSize + i] = variance * 2.0f;
      _policyParameterShifting[_problem->_actionVectorSize + i] = 0.0f;
    }
  }
}

void Continuous::getAction(korali::Sample &sample)
{
 // Get action for all the agents in the environment
 for (size_t i = 0; i < sample["State"].size(); i++)
 {
   // Getting current state
   auto state = sample["State"][i];

   // Adding state to the state time sequence
   _stateTimeSequence.add(state);

   // Storage for the action to select
   std::vector<float> action(_problem->_actionVectorSize);

   // Forward state sequence to get the Gaussian means and sigmas from policy
   auto policy = runPolicy({_stateTimeSequence.getVector()})[0];

   /*****************************************************************************
   * During Training we select action according to policy's probability
   * distribution
   ****************************************************************************/

   if (sample["Mode"] == "Training") action = generateTrainingAction(policy);

   /*****************************************************************************
   * During testing, we select the means (point of highest density) for all
   * elements of the action vector
   ****************************************************************************/

   if (sample["Mode"] == "Testing") action = generateTestingAction(policy);

   /*****************************************************************************
   * Storing the action and its policy
   ****************************************************************************/

   sample["Policy"][i]["Distribution Parameters"] = policy.distributionParameters;
   sample["Policy"][i]["State Value"] = policy.stateValue;
   sample["Policy"][i]["Unbounded Action"] = policy.unboundedAction;
   sample["Action"][i] = action;
 }
}

std::vector<float> Continuous::generateTrainingAction(policy_t &curPolicy)
{
  std::vector<float> action(_problem->_actionVectorSize);

  // Creating the action based on the selected policy
  if (_policyDistribution == "Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mean = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      action[i] = mean + sigma * _normalGenerator->getRandomNumber();
    }
  }

  if (_policyDistribution == "Squashed Normal")
  {
    std::vector<float> unboundedAction(_problem->_actionVectorSize);
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mean = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float scale = _actionScales[i];
      const float shift = _actionShifts[i];
      
      unboundedAction[i] = mean + sigma * _normalGenerator->getRandomNumber();
      action[i] = (std::tanh(unboundedAction[i]) * scale) + shift;
    }
    curPolicy.unboundedAction = unboundedAction;
  }

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      action[i] = ranBetaAlt(_normalGenerator->_range, curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
    }
  }

  return action;
}

std::vector<float> Continuous::generateTestingAction(const policy_t &curPolicy)
{
  std::vector<float> action(_problem->_actionVectorSize);

  if (_policyDistribution == "Normal")
  {
    // Take only the means without noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
      action[i] = curPolicy.distributionParameters[i];
  }

  if (_policyDistribution == "Squashed Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mean = curPolicy.distributionParameters[i];
      const float scale = _actionScales[i];
      const float shift = _actionShifts[i];
      action[i] = (std::tanh(mean) * scale) + shift;
    }
  }

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      action[i] = ranBetaAlt(_normalGenerator->_range, curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
    }
  }

  return action;
}

float Continuous::calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy)
{
  float logpCurPolicy = 0.0f;
  float logpOldPolicy = 0.0f;

  // Storage for importance weight value
  float importanceWeight = 0.0f;

  if (_policyDistribution == "Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      logpOldPolicy += normalLogDensity(action[i], oldMean, oldSigma);
      logpCurPolicy += normalLogDensity(action[i], curMean, curSigma);
    }
  }

  if (_policyDistribution == "Squashed Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      // Importance weight of squashed normal is the importance weight of normal evaluated at unbounded action
      logpOldPolicy += normalLogDensity(oldPolicy.unboundedAction[i], oldMean, oldSigma);
      logpCurPolicy += normalLogDensity(oldPolicy.unboundedAction[i], curMean, curSigma);
    }
  }

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldVariance = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      logpOldPolicy += betaLogDensityAlt(action[i], oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
      logpCurPolicy += betaLogDensityAlt(action[i], curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
    }
  }

  // Calculating log importance weight
  float logImportanceWeight = logpCurPolicy - logpOldPolicy;

  // Normalizing extreme values to prevent loss of precision
  if (logImportanceWeight > +7.0f) logImportanceWeight = +7.0f;
  if (logImportanceWeight < -7.0f) logImportanceWeight = -7.0f;

  // Calculating actual importance weight by exp
  importanceWeight = std::exp(logImportanceWeight);

  return importanceWeight;
}

std::vector<float> Continuous::calculateImportanceWeightGradient(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy)
{
  // Storage for importance weight gradients
  std::vector<float> importanceWeightGradients(2 * _problem->_actionVectorSize, 0.0);

  if (_policyDistribution == "Normal")
  {
    float logpOldPolicy = 0.0f;
    float logpCurPolicy = 0.0f;

    // ParamsOne are the Means, ParamsTwo are the Sigmas
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      // Calculate importance weight
      logpOldPolicy += normalLogDensity(action[i], oldMean, oldSigma);
      logpCurPolicy += normalLogDensity(action[i], curMean, curSigma);

      // Deviation from expAction and current Mean
      float curActionDiff = (action[i] - curMean);

      // Inverse Variances
      float curInvVar = 1. / (curSigma * curSigma);

      // Gradient with respect to Mean
      importanceWeightGradients[i] = curActionDiff * curInvVar;

      // Gradient with respect to Sigma
      importanceWeightGradients[_problem->_actionVectorSize + i] = (curActionDiff * curActionDiff) * (curInvVar / curSigma) - 1.0f / curSigma;
    }

    float logImportanceWeight = logpCurPolicy - logpOldPolicy;
    float importanceWeight = std::exp(logImportanceWeight);

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
      importanceWeightGradients[i] *= importanceWeight;
  }

  if (_policyDistribution == "Squashed Normal")
  {
    float logpOldPolicy = 0.0f;
    float logpCurPolicy = 0.0f;

    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float scale = _actionScales[i];
      const float shift = _actionShifts[i];
      const float unboundedAction = std::atanh((action[i] - shift) / scale);

      // Importance weight of squashed normal is the importance weight of normal evaluated at unbounded action
      logpOldPolicy += normalLogDensity(unboundedAction, oldMean, oldSigma);
      logpCurPolicy += normalLogDensity(unboundedAction, curMean, curSigma);

      // Deviation from expAction and current Mean
      float curActionDiff = (unboundedAction - curMean);

      // Inverse Variances
      float curInvVar = 1. / (curSigma * curSigma);

      // Gradient with respect to Mean
      importanceWeightGradients[i] = curActionDiff * curInvVar;

      // Gradient with respect to Sigma
      importanceWeightGradients[_problem->_actionVectorSize + i] = (curActionDiff * curActionDiff) * (curInvVar / curSigma) - 1.0f / curSigma;
    }

    float logImportanceWeight = logpCurPolicy - logpOldPolicy;
    float importanceWeight = std::exp(logImportanceWeight);

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
      importanceWeightGradients[i] *= importanceWeight;
  }

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldVariance = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      float alphaCur;
      float betaCur;
      std::tie(alphaCur, betaCur) = betaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      float alphaOld;
      float betaOld;
      std::tie(alphaOld, betaOld) = betaParamTransformAlt(oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // Log probability of action with old policy params
      const float logpOldPolicy = betaLogDensityAlt(action[i], oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
      const float invpOldPolicy = std::exp(-logpOldPolicy);

      // Variable preparation
      const float Bab = gsl_sf_beta(alphaCur, betaCur);

      const float psiAb = gsl_sf_psi(alphaCur + betaCur);

      const float actionRange = _actionUpperBounds[i] - _actionLowerBounds[i];
      const float logscale = std::log(actionRange);
      const float powscale = std::pow(actionRange, -betaCur - alphaCur + 1.);
      const float factor = -1. * std::pow(action[i] - _actionLowerBounds[i], alphaCur - 1.) * powscale * std::pow(_actionUpperBounds[i] - action[i], betaCur - 1.) * invpOldPolicy / Bab;

      // Rho Grad wrt alpha and beta
      const float daBab = gsl_sf_psi(alphaCur) - psiAb;
      const float drhoda = ((logscale - std::log(action[i] - _actionLowerBounds[i])) + daBab) * factor;
      const float dbBab = gsl_sf_psi(betaCur) - psiAb;
      const float drhodb = (logscale - std::log(_actionUpperBounds[i] - action[i]) + dbBab) * factor;

      // Derivatives of alpha and beta wrt mu and varc
      float dadmu, dadvarc, dbdmu, dbdvarc;
      std::tie(dadmu, dadvarc, dbdmu, dbdvarc) = derivativesBetaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // Rho Grad wrt mu and varc
      importanceWeightGradients[i] = drhoda * dadmu + drhodb * dbdmu;
      importanceWeightGradients[_problem->_actionVectorSize + i] = drhoda * dadvarc + drhodb * dbdvarc;
    }
  }

  return importanceWeightGradients;
}

std::vector<float> Continuous::calculateKLDivergenceGradient(const policy_t &oldPolicy, const policy_t &curPolicy)
{
  // Storage for KL Divergence Gradients
  std::vector<float> KLDivergenceGradients(2.0 * _problem->_actionVectorSize, 0.0);

  if (_policyDistribution == "Normal" || _policyDistribution == "Squashed Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; ++i)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      float curInvVar = 1. / (curSigma * curSigma);
      float actionDiff = (curMean - oldMean);

      // KL-Gradient with respect to Mean
      KLDivergenceGradients[i] = actionDiff * curInvVar;

      // Contribution to Sigma from Trace
      float gradTr = -(curInvVar / curSigma) * oldSigma * oldSigma;

      // Contribution to Sigma from Quadratic term
      float gradQuad = -(actionDiff * actionDiff) * (curInvVar / curSigma);

      // Contribution to Sigma from Determinant
      float gradDet = 1.0f / curSigma;

      // KL-Gradient with respect to Sigma
      KLDivergenceGradients[_problem->_actionVectorSize + i] = gradTr + gradQuad + gradDet;
    }
  }

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; ++i)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldVariance = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      float alphaCur;
      float betaCur;
      std::tie(alphaCur, betaCur) = betaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      float alphaOld;
      float betaOld;
      std::tie(alphaOld, betaOld) = betaParamTransformAlt(oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      const float psiAbCur = gsl_sf_psi(alphaCur + betaCur);
      const float psiAbOld = gsl_sf_psi(alphaOld + betaOld);

      const float actionRange = _actionUpperBounds[i] - _actionLowerBounds[i];

      // KL Grad wrt alpha
      const float dklda = (gsl_sf_psi(alphaCur) - psiAbCur - gsl_sf_psi(alphaOld) - psiAbOld) / actionRange;

      // KL Grad wrt beta
      const float dkldb = (gsl_sf_psi(betaCur) - psiAbCur - gsl_sf_psi(betaOld) - psiAbOld) / actionRange;

      // Derivatives of alpha and beta wrt mu and varc
      float dadmu, dadvarc, dbdmu, dbdvarc;
      std::tie(dadmu, dadvarc, dbdmu, dbdvarc) = derivativesBetaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // KL Grad wrt mu and varc
      KLDivergenceGradients[i] = dklda * dadmu + dkldb * dbdmu;
      KLDivergenceGradients[_problem->_actionVectorSize + i] = dklda * dadvarc + dkldb * dbdvarc;
    }
  }

  return KLDivergenceGradients;
}

void Continuous::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Normal Generator"))
 {
 _normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(js["Normal Generator"], _k));
 _normalGenerator->applyVariableDefaults();
 _normalGenerator->applyModuleDefaults(js["Normal Generator"]);
 _normalGenerator->setConfiguration(js["Normal Generator"]);
   eraseValue(js, "Normal Generator");
 }

 if (isDefined(js, "Action Shifts"))
 {
 try { _actionShifts = js["Action Shifts"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Action Shifts']\n%s", e.what()); } 
   eraseValue(js, "Action Shifts");
 }

 if (isDefined(js, "Action Scales"))
 {
 try { _actionScales = js["Action Scales"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Action Scales']\n%s", e.what()); } 
   eraseValue(js, "Action Scales");
 }

 if (isDefined(js, "Policy", "Parameter Count"))
 {
 try { _policyParameterCount = js["Policy"]["Parameter Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Count']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Count");
 }

 if (isDefined(js, "Policy", "Parameter Transformation Masks"))
 {
 try { _policyParameterTransformationMasks = js["Policy"]["Parameter Transformation Masks"].get<std::vector<std::string>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Transformation Masks']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Transformation Masks");
 }

 if (isDefined(js, "Policy", "Parameter Scaling"))
 {
 try { _policyParameterScaling = js["Policy"]["Parameter Scaling"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Scaling']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Scaling");
 }

 if (isDefined(js, "Policy", "Parameter Shifting"))
 {
 try { _policyParameterShifting = js["Policy"]["Parameter Shifting"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Shifting']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Shifting");
 }

 if (isDefined(js, "Policy", "Distribution"))
 {
 try { _policyDistribution = js["Policy"]["Distribution"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Distribution']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_policyDistribution == "Normal") validOption = true; 
 if (_policyDistribution == "Squashed Normal") validOption = true; 
 if (_policyDistribution == "Beta") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Policy']['Distribution'] required by continuous.\n", _policyDistribution.c_str()); 
}
   eraseValue(js, "Policy", "Distribution");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Policy']['Distribution'] required by continuous.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Agent::setConfiguration(js);
 _type = "agent/continuous";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: continuous: \n%s\n", js.dump(2).c_str());
} 

void Continuous::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Policy"]["Distribution"] = _policyDistribution;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
   js["Action Shifts"] = _actionShifts;
   js["Action Scales"] = _actionScales;
   js["Policy"]["Parameter Count"] = _policyParameterCount;
   js["Policy"]["Parameter Transformation Masks"] = _policyParameterTransformationMasks;
   js["Policy"]["Parameter Scaling"] = _policyParameterScaling;
   js["Policy"]["Parameter Shifting"] = _policyParameterShifting;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Agent::getConfiguration(js);
} 

void Continuous::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}, \"Policy\": {\"Distribution\": \"Normal\"}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Agent::applyModuleDefaults(js);
} 

void Continuous::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Agent::applyVariableDefaults();
} 

bool Continuous::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Agent::checkTermination();
 return hasFinished;
}



} //agent
} //solver
} //korali
