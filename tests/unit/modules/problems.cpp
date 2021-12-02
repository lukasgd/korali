#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/problem/optimization/optimization.hpp"
#include "modules/problem/sampling/sampling.hpp"
#include "modules/problem/bayesian/bayesian.hpp"
#include "modules/problem/bayesian/custom/custom.hpp"
#include "modules/problem/bayesian/reference/reference.hpp"
#include "modules/problem/hierarchical/hierarchical.hpp"
#include "modules/problem/hierarchical/psi/psi.hpp"
#include "modules/problem/hierarchical/theta/theta.hpp"
#include "modules/problem/hierarchical/thetaNew/thetaNew.hpp"
#include "modules/problem/propagation/propagation.hpp"
#include "modules/problem/integration/integration.hpp"
#include "modules/problem/supervisedLearning/supervisedLearning.hpp"
#include "modules/problem/reinforcementLearning/reinforcementLearning.hpp"
#include "modules/problem/reinforcementLearning/discrete/discrete.hpp"
#include "modules/problem/reinforcementLearning/continuous/continuous.hpp"
#include "sample/sample.hpp"

namespace
{
 using namespace korali;
 using namespace korali::problem;
 using namespace korali::problem::bayesian;
 using namespace korali::problem::hierarchical;

 TEST(Problem, Optimization)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  v._distributionIndex = 0;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Initial Mean"] = 0.0;
  e["Variables"][0]["Initial Standard Deviation"] = 0.25;
  e["Variables"][0]["Lower Bound"] = -1.0;
  e["Variables"][0]["Upper Bound"] = 1.0;
  // Configuring Problem
  e["Problem"]["Type"] = "Optimization";
  Optimization* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Optimization";
  problemJs["Objective Function"] = 0;
  e["Solver"]["Type"] = "Optimizer/CMAES";

  ASSERT_NO_THROW(pObj = dynamic_cast<Optimization *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ 1.0 });
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  pObj->_constraints = std::vector<size_t>({0});

  // Evaluating correct execution of evaluation
  Sample s;
  ASSERT_NO_THROW(pObj->evaluateConstraints(s));
  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateWithGradients(s));

  // Evaluating incorrect execution of evaluation
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::numeric_limits<double>::infinity();
   s["Gradient"] = std::vector<double>(1.0);
  };

  ASSERT_ANY_THROW(pObj->evaluateConstraints(s));
  ASSERT_ANY_THROW(pObj->evaluate(s));
  ASSERT_ANY_THROW(pObj->evaluateWithGradients(s));

  // Evaluating incorrect execution of gradient
  modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ std::numeric_limits<double>::infinity() });
  };

  ASSERT_NO_THROW(pObj->evaluateConstraints(s));
  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_ANY_THROW(pObj->evaluateWithGradients(s));

  // Evaluating incorrect size of gradients
  modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ });
  };

  ASSERT_ANY_THROW(pObj->evaluateWithGradients(s));

  // Evaluating correct execution of multiple evaluations
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::vector<double>({1.0, 1.0});
  };

  ASSERT_NO_THROW(pObj->evaluateMultiple(s));

  // Trying to run unknown operation
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Evaluating incorrect execution of multiple evaluations
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::vector<double>({std::numeric_limits<double>::infinity(), 1.0});
  };

  ASSERT_ANY_THROW(pObj->evaluateMultiple(s));

  // Testing optional parameters
  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Has Discrete Variables"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Has Discrete Variables"] = true;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs.erase("Num Objectives");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Num Objectives"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Num Objectives"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs.erase("Objective Function");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));


  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Objective Function"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Objective Function"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs.erase("Constraints");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Constraints"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Constraints"] = std::vector<uint64_t>({1});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Name");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Name"] = 1.0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Name"] = "Var X";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

 };

 TEST(Problem, SupervisedLearning)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  v._distributionIndex = 0;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Initial Mean"] = 0.0;
  e["Variables"][0]["Initial Standard Deviation"] = 0.25;
  e["Variables"][0]["Lower Bound"] = -1.0;
  e["Variables"][0]["Upper Bound"] = 1.0;

  // Configuring Problem
  SupervisedLearning* pObj;
  knlohmann::json problemJs;

  problemJs["Type"] = "Supervised Learning";
  problemJs["Max Timesteps"] = 1;
  problemJs["Training Batch Size"] = 1;
  problemJs["Inference Batch Size"] = 1;

  problemJs["Input"]["Data"] = std::vector<std::vector<std::vector<float>>>({{{0.0}}});
  problemJs["Input"]["Size"] = 1;
  problemJs["Solution"]["Data"] = std::vector<std::vector<float>>({{0.0}});
  problemJs["Solution"]["Size"] = 1;

  e["Solver"]["Type"] = "Learner/DeepSupervisor";

  ASSERT_NO_THROW(pObj = dynamic_cast<SupervisedLearning *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing data verifications
  ASSERT_NO_THROW(pObj->verifyData());
  pObj->_inputData = std::vector<std::vector<std::vector<float>>>({{{0.0, 0.0}}});
  ASSERT_ANY_THROW(pObj->verifyData());
  pObj->_inputData = std::vector<std::vector<std::vector<float>>>({{{0.0}, {0.0}}});
  ASSERT_ANY_THROW(pObj->verifyData());
  pObj->_inputData = std::vector<std::vector<std::vector<float>>>({{{0.0}}, {{0.0}}});
  ASSERT_ANY_THROW(pObj->verifyData());
  pObj->_inputData = std::vector<std::vector<std::vector<float>>>({{{0.0}}});
  ASSERT_NO_THROW(pObj->verifyData());
  pObj->_solutionData = std::vector<std::vector<float>>({{0.0, 0.0}});
  ASSERT_ANY_THROW(pObj->verifyData());
  pObj->_solutionData = std::vector<std::vector<float>>({{0.0}, {0.0}});
  ASSERT_ANY_THROW(pObj->verifyData());

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Training Batch Size");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Training Batch Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Training Batch Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Inference Batch Size");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Inference Batch Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Inference Batch Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Max Timesteps");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Max Timesteps"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Max Timesteps"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"].erase("Data");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"]["Data"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"]["Data"] = std::vector<std::vector<std::vector<float>>>({{{0.0}}});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"].erase("Size");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"]["Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"]["Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"].erase("Data");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"]["Data"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"]["Data"] = std::vector<std::vector<float>>({{0.0}});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"].erase("Size");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"]["Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"]["Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, Sampling)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  v._distributionIndex = 0;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Initial Mean"] = 0.0;
  e["Variables"][0]["Initial Standard Deviation"] = 0.25;
  e["Variables"][0]["Lower Bound"] = -1.0;
  e["Variables"][0]["Upper Bound"] = 1.0;

  // Configuring Problem
  e["Problem"]["Type"] = "Sampling";
  Sampling* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Sampling";
  problemJs["Probability Function"] = 0;
  e["Solver"]["Type"] = "Sampler/MCMC";

  ASSERT_NO_THROW(pObj = dynamic_cast<Sampling *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  // Evaluating correct execution of evaluation
  Sample s;
  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateGradient(s));
  ASSERT_NO_THROW(pObj->evaluateHessian(s));

  // Trying to run unknown operation
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // running correct operations
  ASSERT_NO_THROW(pObj->runOperation("Evaluate", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Gradient", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Hessian", s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logP(x)"] = std::numeric_limits<double>::infinity();
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_ANY_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateGradient(s));
  ASSERT_NO_THROW(pObj->evaluateHessian(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_ANY_THROW(pObj->evaluateGradient(s));
  ASSERT_NO_THROW(pObj->evaluateHessian(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>(2);
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateHessian(s));

  modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{}});
  };
  _functionVector[0] = &modelFc;
  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateHessian(s));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Probability Function");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Probability Function"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Probability Function"] = modelFc;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, Propagation)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);

  // Configuring Problem
  Propagation* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Propagation";
  problemJs["Execution Model"] = 0;
  e["Solver"]["Type"] = "Executor";

  ASSERT_NO_THROW(pObj = dynamic_cast<Propagation *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Triggering error on variable with distribution and distributions defined
  pObj->_numberOfSamples = 0;
  pObj->initialize();
  ASSERT_NO_THROW(pObj->initialize());

  pObj->_numberOfSamples = 10;
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_numberOfSamples = 0;
  v._priorDistribution = "Uniform";
  ASSERT_ANY_THROW(pObj->initialize());

  // Testing configuration

  baseExpJs["Variables"][0]["Name"] = "Var 1";
  baseExpJs["Variables"][0]["Precomputed Values"] = std::vector<double>({0.0});
  baseExpJs["Variables"][0]["Prior Distribution"] = "Uniform";
  baseExpJs["Variables"][0]["Distribution Index"] = 0;
  baseExpJs["Variables"][0]["Sampled Values"] = std::vector<double>({0.0});

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Execution Model");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Execution Model"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Execution Model"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Number Of Samples");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Number Of Samples"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Number Of Samples"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Precomputed Values");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Precomputed Values"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Precomputed Values"] = std::vector<double>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Prior Distribution");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = 0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = "Uniform";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Distribution Index");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Sampled Values");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sampled Values"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sampled Values"] = std::vector<double>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, Integration)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);

  // Integrand function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s) { s["Evaluation"] = 1.0; };

  // Configuring Problem
  Integration* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Integration";
  problemJs["Integrand"] = modelFc;
  problemJs["Integration Method"] = "Rectangle";
  e["Solver"]["Type"] = "Integrator";

  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Number Of Gridpoints"] = 10;
  e["Variables"][0]["Sampling Distribution"] = "Uniform";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;
  e["Variables"][0]["Sample Points"] = std::vector<double>({ 0.0, 0.1, 0.2, 0.3 });
  e["Variables"][0]["Quadrature Weights"] = std::vector<double>({ 0.0, 0.1, 0.2, 0.3 });

  ASSERT_NO_THROW(pObj = dynamic_cast<Integration *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  ASSERT_NO_THROW(pObj->initialize());

  // Testing fail cases
  pObj->_integrationMethod = "Monte Carlo";
  e._variables[0]->_samplingDistribution = "Unknown";
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_integrationMethod = "Custom";
  e._variables[0]->_samplingDistribution = "";
  e._variables[0]->_quadratureWeights = std::vector<double>({});
  e._variables[0]->_samplePoints = std::vector<double>({0.0});
  ASSERT_ANY_THROW(pObj->initialize());

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Integrand");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integrand"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integrand"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Integration Method");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integration Method"] = 1.0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integration Method"] = "Unknown";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integration Method"] = "Rectangle";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Lower Bound");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Lower Bound"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Lower Bound"] = 1.0;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Upper Bound");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Upper Bound"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Upper Bound"] = 1.0;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Number Of Gridpoints");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Number Of Gridpoints"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Number Of Gridpoints"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Sampling Distribution");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sampling Distribution"] = 1.0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sampling Distribution"] = "Uniform";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Distribution Index");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Sample Points");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sample Points"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sample Points"] = std::vector<double>({});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Quadrature Weights");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Quadrature Weights"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Quadrature Weights"] = std::vector<double>({});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, BayesianCustom)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);

  // Configuring Problem
  Custom* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Bayesian/Custom";
  problemJs["Likelihood Model"] = 0;
  e["Solver"]["Type"] = "Sampler/TMCMC";

  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Number Of Gridpoints"] = 10;
  e["Variables"][0]["Prior Distribution"] = "Uniform";

  ASSERT_NO_THROW(pObj = dynamic_cast<Custom *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  ASSERT_NO_THROW(pObj->initialize());

  // Testing initialization fail
  v._priorDistribution = "Unknown";
  ASSERT_ANY_THROW(pObj->initialize());
  v._priorDistribution = "Uniform";

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["logLikelihood"] = 0.1;
   s["logLikelihood Gradient"] = std::vector<double>({0.1});
   s["Fisher Information"] = std::vector<std::vector<double>>({{0.1}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateFisherInformation(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s._js.getJson().erase("logLikelihood");
   s._js.getJson().erase("logLikelihood Gradient");
   s._js.getJson().erase("Fisher Information");
  };

  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateFisherInformation(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logLikelihood"] = 0.1;
   s["logLikelihood Gradient"] = std::vector<double>();
   s["Fisher Information"] = std::vector<std::vector<double>>({{0.1}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  pObj->evaluateLoglikelihood(s);
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateFisherInformation(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logLikelihood"] = 0.1;
   s["logLikelihood Gradient"] = std::vector<double>({0.0});
   s["Fisher Information"] = std::vector<std::vector<double>>();
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  pObj->evaluateLoglikelihood(s);
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateFisherInformation(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logLikelihood"] = 0.1;
   s["logLikelihood Gradient"] = std::vector<double>({0.0});
   s["Fisher Information"] = std::vector<std::vector<double>>({});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  pObj->evaluateLoglikelihood(s);
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateFisherInformation(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Likelihood Model");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Prior Distribution");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = 1;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = "Uniform";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Distribution Index");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, BayesianReference)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);

  // Configuring Problem
  Reference* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Bayesian/Reference";
  problemJs["Computational Model"] = 0;
  problemJs["Likelihood Model"] = "Normal";
  problemJs["Reference Data"] = std::vector<double>({ 0.0 });
  e["Solver"]["Type"] = "Sampler/TMCMC";

  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Number Of Gridpoints"] = 10;
  e["Variables"][0]["Prior Distribution"] = "Uniform";

  ASSERT_NO_THROW(pObj = dynamic_cast<Reference *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  s["Sample Id"] = 0;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  ASSERT_NO_THROW(pObj->initialize());

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  pObj->_likelihoodModel = "Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "StudentT";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Poisson";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Geometric";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s._js.getJson().erase("Standard Deviation");
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  pObj->_likelihoodModel = "Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s._js.getJson().erase("Reference Evaluations");
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Poisson";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Geometric";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Unknown";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  ASSERT_ANY_THROW(pObj->evaluateFisherInformation(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>();
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Poisson";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Geometric";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>();
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s._js.getJson().erase("Degrees Of Freedom");
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>();
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s._js.getJson().erase("Dispersion");
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>();
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  // Gradient evaluation
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::numeric_limits<double>::infinity();
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});

   s["Gradient Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Dispersion"] = std::vector<std::vector<double>>({{0.1}});

   s["Hessian Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Dispersion"] = std::vector<std::vector<double>>({{0.1}});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));
  ASSERT_NO_THROW(pObj->evaluateFisherInformation(s));

  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));
  ASSERT_NO_THROW(pObj->evaluateFisherInformation(s));

  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));
  ASSERT_NO_THROW(pObj->evaluateFisherInformation(s));

  modelFc = [](Sample& s)
  {
   s["F(x)"] = 0.1;
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});

   s["Gradient Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Dispersion"] = std::vector<std::vector<double>>({{0.1}});

   s["Hessian Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Dispersion"] = std::vector<std::vector<double>>({{0.1}});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));

  // Running operations
  ASSERT_NO_THROW(pObj->runOperation("Evaluate", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate logPrior", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate logLikelihood", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate logPosterior", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Gradient", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Hessian", s));

  pObj->_likelihoodModel = "Normal";
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Fisher Information", s));
  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Fisher Information", s));
  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_ANY_THROW(pObj->runOperation("Evaluate Fisher Information", s));
  pObj->_likelihoodModel = "Unknown";
  ASSERT_ANY_THROW(pObj->runOperation("Evaluate Fisher Information", s));



  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "StudentT";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "Poisson";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "Geometric";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));

  modelFc = [](Sample& s)
  {
   s["F(x)"] = 0.1;
   s["Reference Evaluations"] = std::vector<double>({-0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});

   s["Gradient Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Dispersion"] = std::vector<std::vector<double>>({{0.1}});

   s["Hessian Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Dispersion"] = std::vector<std::vector<double>>({{0.1}});
  };

  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_referenceData[0] = -1.0;
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Computational Model");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Computational Model"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Computational Model"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Reference Data");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Reference Data"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Reference Data"] = std::vector<double>({0.0});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Likelihood Model");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = "Unknown";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = 1;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = "Normal";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  pObj->_likelihoodModel = "Positive Normal";
  e._variables.push_back(&v);
  pObj->_referenceData = std::vector<double>({0.5, 0.05});
  s["Reference Evaluations"] = std::vector<double>({0.5, 0.05});
  s["Standard Deviation"] = std::vector<double>({0.5, 0.05});
  s["Gradient Mean"] = std::vector<std::vector<double>>({{0.5, 0.05}, {0.5, 0.05}});
  s["Gradient Standard Deviation"] = std::vector<std::vector<double>>({{0.5, 0.05}, {0.5, 0.05}});
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Fisher Information", s));
 }

 TEST(Problem, HierarchicalPsi)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);
  e["Solver"]["Type"] = "Sampler/TMCMC";
  e["Solver"]["Sample Database"] = std::vector<std::vector<double>>({{0.0}});
  e["Solver"]["Sample LogPrior Database"] = std::vector<double>({0.0});
  e["Solver"]["Sample LogLikelihood Database"] = std::vector<double>({0.0});
  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Prior Distribution"] = "Uniform";

  // Configuring Problem
  Psi* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Hierarchical/Psi";
  problemJs["Sub Experiments"] = std::vector<knlohmann::json>({e._js.getJson(), e._js.getJson()});
  problemJs["Conditional Priors"] = std::vector<std::string>({"Uniform"});

  ASSERT_NO_THROW(pObj = dynamic_cast<Psi *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  ASSERT_NO_THROW(pObj->initialize());

  pObj->_conditionalPriors[0] = "Undefined";
  ASSERT_ANY_THROW(pObj->initialize());
  pObj->_conditionalPriors[0] = "Uniform";

  e._variables[0]->_priorDistribution = "Unknown";
  ASSERT_ANY_THROW(pObj->Hierarchical::initialize());
  e._variables[0]->_priorDistribution = "Uniform";

  s["Parameters"][0] = std::numeric_limits<double>::infinity();
  ASSERT_FALSE(pObj->Hierarchical::isSampleFeasible(s));
  s["Parameters"][0] = 0.5;
  ASSERT_TRUE(pObj->Hierarchical::isSampleFeasible(s));

  pObj->_subExperiments[0]["Variables"] = std::vector<knlohmann::json>();
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_subExperiments[0]["Variables"][0]["Name"] = "Var X";
  pObj->_subExperiments[0]["Variables"][0]["Prior Distribution"] = "Uniform";
  pObj->_subExperiments[0]["Solver"]["Sample LogPrior Database"] = std::vector<double>({std::numeric_limits<double>::infinity()});
  ASSERT_ANY_THROW(pObj->initialize());
  pObj->_subExperiments[0]["Solver"]["Sample LogPrior Database"] = std::vector<double>({0.0});

  pObj->_subExperiments[0]["Solver"].erase("Sample LogPrior Database");
  ASSERT_ANY_THROW(pObj->initialize());
  pObj->_subExperiments[0]["Solver"]["Sample LogPrior Database"] = std::vector<double>({0.0});

  pObj->_subExperiments[0]["Is Finished"] = false;
  ASSERT_ANY_THROW(pObj->initialize());

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Conditional Priors");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Conditional Priors"] = 1.0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Conditional Priors"] = std::vector<knlohmann::json>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Sub Experiments");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Sub Experiments"] = 1.0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Sub Experiments"] = std::vector<knlohmann::json>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Prior Distribution");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = 1.0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = "Uniform";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Distribution Index");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, HierarchicalTheta)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);
  e["Solver"]["Type"] = "Sampler/TMCMC";
  e["Solver"]["Sample Database"] = std::vector<std::vector<double>>({{0.0}});
  e["Solver"]["Sample LogPrior Database"] = std::vector<double>({0.0});
  e["Solver"]["Sample LogLikelihood Database"] = std::vector<double>({0.0});
  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Prior Distribution"] = "Uniform";

  knlohmann::json psiExp;
  psiExp["Is Finished"] = true;
  psiExp["Problem"]["Type"] = "Hierarchical/Psi";
  psiExp["Problem"]["Sub Experiments"] = std::vector<knlohmann::json>({e._js.getJson(), e._js.getJson()});
  psiExp["Problem"]["Conditional Priors"] = std::vector<std::string>({"Uniform"});
  psiExp["Variables"][0]["Name"] = "Var X";
  psiExp["Variables"][0]["Prior Distribution"] = "Uniform";
  psiExp["Distributions"][0]["Name"] = "Uniform";
  psiExp["Distributions"][0]["Type"] = "Univariate/Uniform";
  psiExp["Distributions"][0]["Minimum"] = 0.0;
  psiExp["Distributions"][0]["Maximum"] = 1.0;
  psiExp["Solver"]["Type"] = "Sampler/TMCMC";
  psiExp["Solver"]["Population Size"] = 1000;
  psiExp["Solver"]["Burn In"] = 3;
  psiExp["Solver"]["Target Coefficient Of Variation"] = 0.6;
  psiExp["Solver"]["Covariance Scaling"] = 0.01;
  psiExp["Solver"]["Sample Database"] = std::vector<std::vector<double>>({{0.1}});
  psiExp["Solver"]["Sample LogPrior Database"] = std::vector<double>({0.1});
  psiExp["Solver"]["Sample LogLikelihood Database"] = std::vector<double>({0.1});
  psiExp["Solver"]["Chain Leaders LogLikelihoods"] = std::vector<double>({0.1});

  knlohmann::json subExp;
  subExp["Is Finished"] = true;
  subExp["Problem"]["Type"] = "Bayesian/Reference";
  subExp["Problem"]["Reference Data"] = std::vector<double>({0.0});
  subExp["Problem"]["Likelihood Model"] = "Normal";
  subExp["Problem"]["Computational Model"] = 0;
  subExp["Variables"][0]["Name"] = "Var X";
  subExp["Variables"][0]["Prior Distribution"] = "Uniform";
  subExp["Distributions"][0]["Name"] = "Uniform";
  subExp["Distributions"][0]["Type"] = "Univariate/Uniform";
  subExp["Distributions"][0]["Minimum"] = 0.0;
  subExp["Distributions"][0]["Maximum"] = 1.0;
  subExp["Solver"]["Type"] = "Sampler/TMCMC";
  subExp["Solver"]["Population Size"] = 1000;
  subExp["Solver"]["Burn In"] = 3;
  subExp["Solver"]["Target Coefficient Of Variation"] = 0.6;
  subExp["Solver"]["Covariance Scaling"] = 0.01;
  subExp["Solver"]["Sample Database"] = std::vector<std::vector<double>>({{0.1}});
  subExp["Solver"]["Sample LogPrior Database"] = std::vector<double>({0.1});
  subExp["Solver"]["Sample LogLikelihood Database"] = std::vector<double>({0.1});
  subExp["Solver"]["Chain Leaders LogLikelihoods"] = std::vector<double>({0.1});

  // Configuring Problem
  Theta* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Hierarchical/Theta";
  problemJs["Sub Experiment"] = subExp;
  problemJs["Psi Experiment"] = psiExp;

  ASSERT_NO_THROW(pObj = dynamic_cast<Theta *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  pObj->_psiExperiment = psiExp;
  pObj->_subExperiment = subExp;
  ASSERT_NO_THROW(pObj->initialize());

  knlohmann::json optExp;
  optExp["Variables"][0]["Name"] = "Var 1";
  optExp["Variables"][0]["Lower Bound"] = -1.0;
  optExp["Variables"][0]["Upper Bound"] = 1.0;
  optExp["Problem"]["Type"] = "Optimization";
  optExp["Problem"]["Objective Function"] = 0;
  optExp["Solver"]["Type"] = "Optimizer/CMAES";
  optExp["Solver"]["Population Size"] = 16;
  pObj->_psiExperiment = optExp;
  pObj->_subExperiment = subExp;
  ASSERT_ANY_THROW(pObj->initialize());

  optExp["Variables"][0]["Name"] = "Var 1";
  optExp["Variables"][0]["Lower Bound"] = -1.0;
  optExp["Variables"][0]["Upper Bound"] = 1.0;
  optExp["Problem"]["Type"] = "Optimization";
  optExp["Problem"]["Objective Function"] = 0;
  optExp["Solver"]["Type"] = "Optimizer/CMAES";
  optExp["Solver"]["Population Size"] = 16;
  pObj->_psiExperiment = psiExp;
  pObj->_subExperiment = optExp;
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_subExperiment = subExp;
  pObj->_psiExperiment["Problem"]["Conditional Priors"] = std::vector<std::string>({"Uniform", "Uniform"});
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_subExperiment = subExp;
  pObj->_subExperiment["Is Finished"] = false;
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_subExperiment = subExp;
  pObj->_psiExperiment["Problem"]["Type"] = "Sampling";
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_subExperiment = subExp;
  pObj->_psiExperiment["Problem"]["Conditional Priors"] = std::vector<double>({});
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_subExperiment = subExp;
  pObj->_psiExperiment["Solver"]["Sample LogPrior Database"] = std::vector<double>({std::numeric_limits<double>::infinity()});
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_subExperiment = subExp;
  pObj->_subExperiment["Solver"]["Sample LogPrior Database"] = std::vector<double>({std::numeric_limits<double>::infinity()});
  ASSERT_ANY_THROW(pObj->initialize());

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Sub Experiment");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Sub Experiment"] = std::vector<knlohmann::json>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Psi Experiment");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Psi Experiment"] = std::vector<knlohmann::json>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, HierarchicalThetaNew)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);
  e["Solver"]["Type"] = "Sampler/TMCMC";
  e["Solver"]["Sample Database"] = std::vector<std::vector<double>>({{0.0}});
  e["Solver"]["Sample LogPrior Database"] = std::vector<double>({0.0});
  e["Solver"]["Sample LogLikelihood Database"] = std::vector<double>({0.0});
  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Prior Distribution"] = "Uniform";

  knlohmann::json psiExp;
  psiExp["Is Finished"] = true;
  psiExp["Problem"]["Type"] = "Hierarchical/Psi";
  psiExp["Problem"]["Sub Experiments"] = std::vector<knlohmann::json>({e._js.getJson(), e._js.getJson()});
  psiExp["Problem"]["Conditional Priors"] = std::vector<std::string>({"Uniform"});
  psiExp["Variables"][0]["Name"] = "Var X";
  psiExp["Variables"][0]["Prior Distribution"] = "Uniform";
  psiExp["Distributions"][0]["Name"] = "Uniform";
  psiExp["Distributions"][0]["Type"] = "Univariate/Uniform";
  psiExp["Distributions"][0]["Minimum"] = 0.0;
  psiExp["Distributions"][0]["Maximum"] = 1.0;
  psiExp["Solver"]["Type"] = "Sampler/TMCMC";
  psiExp["Solver"]["Population Size"] = 1000;
  psiExp["Solver"]["Burn In"] = 3;
  psiExp["Solver"]["Target Coefficient Of Variation"] = 0.6;
  psiExp["Solver"]["Covariance Scaling"] = 0.01;
  psiExp["Solver"]["Sample Database"] = std::vector<std::vector<double>>({{0.1}});
  psiExp["Solver"]["Sample LogPrior Database"] = std::vector<double>({0.1});
  psiExp["Solver"]["Sample LogLikelihood Database"] = std::vector<double>({0.1});
  psiExp["Solver"]["Chain Leaders LogLikelihoods"] = std::vector<double>({0.1});

  // Configuring Problem
  ThetaNew* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Hierarchical/ThetaNew";
  problemJs["Psi Experiment"] = psiExp;

  ASSERT_NO_THROW(pObj = dynamic_cast<ThetaNew *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  pObj->_psiExperiment = psiExp;
  ASSERT_NO_THROW(pObj->initialize());

  s["Parameters"] = std::vector<double>({0.0});
  s["Sample Id"] = 0;
  s["LogPrior"] = 0.5;
  s["LogLikelihood"] = 0.5;
  s["LogPosterior"] = 0.5;
  ASSERT_NO_THROW(pObj->runOperation("Check Feasibility", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate logPrior", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate logLikelihood", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate logPosterior", s));

  pObj->_psiExperiment = psiExp;
  pObj->_psiExperiment["Is Finished"] = false;
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_psiExperiment["Problem"]["Conditional Priors"] = std::vector<std::string>();
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_psiExperiment["Problem"]["Conditional Priors"] = std::vector<std::string>({"Uniform", "Uniform"});
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_psiExperiment["Solver"]["Sample LogPrior Database"] = std::vector<double>({std::numeric_limits<double>::infinity()});
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_psiExperiment = psiExp;
  pObj->_psiExperiment["Problem"]["Conditional Priors"] = std::vector<std::string>({"Uniform", "Uniform"});
  ASSERT_ANY_THROW(pObj->initialize());

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Psi Experiment");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Psi Experiment"] = std::vector<knlohmann::json>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, ReinforcementLearningDiscrete)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  e["Variables"][0]["Name"] = "X";
  e["Variables"][0]["Type"] = "State";
  e["Variables"][1]["Name"] = "Y";
  e["Variables"][1]["Type"] = "Action";
  e["Variables"][1]["Initial Exploration Noise"] = 0.45;
  e["Solver"]["Type"] = "Agent / Discrete / dVRACER";

  Variable v1;
  Variable v2;
  e._variables.push_back(&v1);
  e._variables.push_back(&v2);
  e._variables[0]->_name = "X";
  e._variables[0]->_type = "State";
  e._variables[1]->_name = "Y";
  e._variables[1]->_type = "Action";
  e._variables[1]->_initialExplorationNoise = 0.45;

  // Configuring Problem
  reinforcementLearning::Discrete* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Reinforcement Learning / Discrete";
  problemJs["Environment Function"] = 0;
  problemJs["Possible Actions"] = std::vector<std::vector<float>>({{0.0}, {1.0}});

  ASSERT_NO_THROW(pObj = dynamic_cast<reinforcementLearning::Discrete *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->ReinforcementLearning::setConfiguration(problemJs));
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing correct initialize
  ASSERT_NO_THROW(pObj->initialize());

  pObj->_possibleActions = std::vector<std::vector<float>>();
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_possibleActions = std::vector<std::vector<float>>({{0.0, 0.1}, {1.0, 1.1}});
  ASSERT_ANY_THROW(pObj->initialize());

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Action Vector Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Action Vector Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["State Vector Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["State Vector Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Action Vector Indexes"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Action Vector Indexes"] = std::vector<size_t>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["State Vector Indexes"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["State Vector Indexes"] = std::vector<size_t>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Agents Per Environment");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Agents Per Environment"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Agents Per Environment"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Environment Function");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Environment Function"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Environment Function"] = 0;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Possible Actions");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Possible Actions"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Possible Actions"] = std::vector<std::vector<float>>({{0.0, 0.1}, {1.0, 1.1}});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Actions Between Policy Updates");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Actions Between Policy Updates"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Actions Between Policy Updates"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Custom Settings");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Custom Settings"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Type");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Type"] = 1;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Type"] = "State";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Lower Bound");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Lower Bound"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Lower Bound"] = 1.0;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Upper Bound");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Upper Bound"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Upper Bound"] = 1.0;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, ReinforcementLearningContinuous)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   e["Variables"][0]["Name"] = "X";
   e["Variables"][0]["Type"] = "State";
   e["Variables"][1]["Name"] = "Y";
   e["Variables"][1]["Type"] = "Action";
   e["Variables"][1]["Initial Exploration Noise"] = 0.45;
   e["Solver"]["Type"] = "Agent / Continuous / VRACER";

   Variable v1;
   Variable v2;
   e._variables.push_back(&v1);
   e._variables.push_back(&v2);
   e._variables[0]->_name = "X";
   e._variables[0]->_type = "State";
   e._variables[1]->_name = "Y";
   e._variables[1]->_type = "Action";
   e._variables[1]->_initialExplorationNoise = 0.45;

   // Configuring Problem
   reinforcementLearning::Continuous* pObj;
   knlohmann::json problemJs;
   problemJs["Type"] = "Reinforcement Learning / Continuous";
   problemJs["Environment Function"] = 0;

   ASSERT_NO_THROW(pObj = dynamic_cast<reinforcementLearning::Continuous *>(Module::getModule(problemJs, &e)));
   e._problem = pObj;

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(pObj->applyVariableDefaults());

   // Trying to run unknown operation
   Sample s;
   ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

   // Backup the correct base configuration
   auto baseProbJs = problemJs;
   auto baseExpJs = experimentJs;

   // Testing correct configuration
   ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

   // Testing unrecognized solver
   problemJs = baseProbJs;
   experimentJs = baseExpJs;
   e["Solver"]["Type"] = "";
   ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

   // Testing correct initialize
   ASSERT_NO_THROW(pObj->initialize());
  }
} // namespace
