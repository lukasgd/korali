#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/optimizer/Adam/Adam.hpp"
#include "modules/solver/optimizer/AdaBelief/AdaBelief.hpp"
#include "modules/solver/optimizer/DEA/DEA.hpp"
#include "modules/solver/optimizer/CMAES/CMAES.hpp"
#include "modules/solver/optimizer/MOCMAES/MOCMAES.hpp"
#include "modules/solver/optimizer/MADGRAD/MADGRAD.hpp"
#include "modules/solver/optimizer/Rprop/Rprop.hpp"
#include "modules/solver/optimizer/gridSearch/gridSearch.hpp"
#include "modules/problem/optimization/optimization.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::solver::optimizer;
 using namespace korali::problem;

 //////////////// BASE CLASS ////////////////////////

  TEST(optimizers, baseClass)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   Variable v;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Lower Bound"] = 0.0;
   e["Variables"][0]["Upper Bound"] = 1.0;

   // Creating optimizer configuration Json
   knlohmann::json optimizerJs;
   optimizerJs["Type"] = "Optimizer/AdaBelief";

   // Creating module
   AdaBelief* opt;
   ASSERT_NO_THROW(opt = dynamic_cast<AdaBelief *>(Module::getModule(optimizerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(opt->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = optimizerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   // Testing optional parameters

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Current Best Value"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Current Best Value"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Previous Best Value"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Previous Best Value"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Best Ever Value"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Best Ever Value"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Best Ever Variables"] = std::vector<double>({ 2.0 });
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Best Ever Variables"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"]["Max Value"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"].erase("Max Value");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"]["Max Value"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"]["Min Value Difference Threshold"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"].erase("Min Value Difference Threshold");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"]["Min Value Difference Threshold"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Lower Bound");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Lower Bound"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Lower Bound"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Upper Bound");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Upper Bound"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Upper Bound"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Mean");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Mean"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Mean"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Standard Deviation");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Standard Deviation"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Standard Deviation"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Minimum Standard Deviation Update");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Minimum Standard Deviation Update"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Minimum Standard Deviation Update"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Initial Value");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Value"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Initial Value"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0].erase("Values");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Values"] = "Not a Number";
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   e["Variables"][0]["Values"] = std::vector<double>({});
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));
  }

 //////////////// ADABELIEF ////////////////////////

 TEST(optimizers, AdaBelief)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;

  // Creating optimizer configuration Json
  knlohmann::json optimizerJs;
  optimizerJs["Type"] = "Optimizer/AdaBelief";

  // Creating module
  AdaBelief* opt;
  ASSERT_NO_THROW(opt = dynamic_cast<AdaBelief *>(Module::getModule(optimizerJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(opt->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = optimizerJs;
  auto baseExpJs = experimentJs;

  // Setting up optimizer correctly
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing infinite initial value fail
  v._initialValue = std::numeric_limits<double>::infinity();
  ASSERT_ANY_THROW(opt->setInitialConfiguration());

  // Testing initial configuration success
  v._initialValue = 1.0;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing optional parameters
  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Variable"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Variable"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Norm"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Norm"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["First Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["First Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected First Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected First Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Second Central Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Second Central Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected Second Central Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected Second Central Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta1"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Beta1");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta1"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta2"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Beta2");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta2"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Eta"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Eta");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Eta"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Epsilon"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Epsilon");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Epsilon"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Gradient Norm"] = std::vector<double>({ 1.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Gradient Norm");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Gradient Norm"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Gradient Norm"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Gradient Norm");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Gradient Norm"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing termination criteria
  e._currentGeneration = 2;
  opt->_gradientNorm = 0.0;
  ASSERT_TRUE(opt->checkTermination());
  opt->_gradientNorm = 1.5;
  ASSERT_FALSE(opt->checkTermination());
  opt->_gradientNorm = 3.0;
  ASSERT_TRUE(opt->checkTermination());
 }

 //////////////// ADAM ////////////////////////

 TEST(optimizers, Adam)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;

  // Creating optimizer configuration Json
  knlohmann::json optimizerJs;
  optimizerJs["Type"] = "Optimizer/Adam";

  // Creating module
  Adam* opt;
  ASSERT_NO_THROW(opt = dynamic_cast<Adam *>(Module::getModule(optimizerJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(opt->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = optimizerJs;
  auto baseExpJs = experimentJs;

  // Setting up optimizer correctly
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing infinite initial value fail
  v._initialValue = std::numeric_limits<double>::infinity();
  ASSERT_ANY_THROW(opt->setInitialConfiguration());

  // Testing initial configuration success
  v._initialValue = 1.0;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing optional parameters
  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Variable"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Variable"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Squared Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Squared Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Norm"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Norm"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["First Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["First Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected First Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected First Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Second Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Second Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected Second Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected Second Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta1"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Beta1");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta1"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta2"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Beta2");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta2"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Eta"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Eta");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Eta"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Epsilon"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Epsilon");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Epsilon"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Gradient Norm"] = std::vector<double>({ 1.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Gradient Norm");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Gradient Norm"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Gradient Norm"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Gradient Norm");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Gradient Norm"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing termination criteria
  e._currentGeneration = 2;
  opt->_gradientNorm = 0.0;
  ASSERT_TRUE(opt->checkTermination());
  opt->_gradientNorm = 1.5;
  ASSERT_FALSE(opt->checkTermination());
  opt->_gradientNorm = 3.0;
  ASSERT_TRUE(opt->checkTermination());
 }

 //////////////// CMAES ////////////////////////

 TEST(optimizers, CMAES)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;
  e["Problem"]["Type"] = "Optimization";
  e["Problem"]["Constraints"][0] = 0;

  // Creating problem module
  Optimization* p;
  knlohmann::json problemJs;
  problemJs["Type"] = "Optimization";
  ASSERT_NO_THROW(p = dynamic_cast<Optimization *>(Module::getModule(problemJs, &e)));
  e._problem = p;
  p->_numObjectives = 2;

  // Creating optimizer configuration Json
  knlohmann::json optimizerJs;
  optimizerJs["Type"] = "Optimizer/CMAES";
  optimizerJs["Population Size"] = 4;

  // Creating module
  CMAES* opt;
  ASSERT_NO_THROW(opt = dynamic_cast<CMAES *>(Module::getModule(optimizerJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(opt->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = optimizerJs;
  auto baseExpJs = experimentJs;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing initial configuration failures
  opt->_globalSuccessLearningRate = -1.0;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  opt->_globalSuccessLearningRate = 2.0;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  opt->_globalSuccessLearningRate = 0.5;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  opt->_targetSuccessRate  = -1.0;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  opt->_targetSuccessRate  = 2.0;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  opt->_targetSuccessRate  = 0.5;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  opt->_covarianceMatrixAdaptionStrength   = -1.0;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  opt->_covarianceMatrixAdaptionStrength   = 0.5;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing optional parameters
  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Viability Regime"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Viability Regime"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Value Vector"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Value Vector"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradients"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradients"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Population Size"] = 2;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Population Size"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mu Value"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mu Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Weights"] = std::vector<double>({1.0, 1.0, 1.0, 1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Weights"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Effective Mu"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Effective Mu"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sigma Cumulation Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sigma Cumulation Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Damp Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Damp Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Cumulative Covariance"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Cumulative Covariance"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Chi Square Number"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Chi Square Number"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Eigenvalue Evaluation Frequency"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Eigenvalue Evaluation Frequency"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sigma"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sigma"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Trace"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Trace"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Population"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Population"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Finished Sample Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Finished Sample Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variables"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variables"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Value"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Ever Value"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Ever Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sorting Index"] = std::vector<size_t>(1.0);
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sorting Index"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Covariance Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Covariance Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Eigenvector Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Eigenvector Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Covariance Eigenvector Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Covariance Eigenvector Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Axis Lengths"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Axis Lengths"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Axis Lengths"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Axis Lengths"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["BDZ Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["BDZ Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar BDZ Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar BDZ Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mean"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mean"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Mean"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Mean"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mean Update"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mean Update"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Evolution Path"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Evolution Path"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Conjugate Evolution Path"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Conjugate Evolution Path"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Conjugate Evolution Path L2 Norm"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Conjugate Evolution Path L2 Norm"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Infeasible Sample Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Infeasible Sample Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Maximum Diagonal Covariance Matrix Element"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Maximum Diagonal Covariance Matrix Element"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Minimum Diagonal Covariance Matrix Element"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Minimum Diagonal Covariance Matrix Element"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Maximum Covariance Eigenvalue"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Maximum Covariance Eigenvalue"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Minimum Covariance Eigenvalue"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Minimum Covariance Eigenvalue"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Eigensystem Updated"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Eigensystem Updated"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Indicator"] = std::vector<std::vector<int>>({{1}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Indicator"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaption Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaption Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Valid Sample"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Valid Sample"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Global Success Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Global Success Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Function Value"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Function Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Resampled Parameter Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Resampled Parameter Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaptation Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaptation Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Boundaries"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Boundaries"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Improvement"] = std::vector<int>({1});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Improvement"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Constraint Violation Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Constraint Violation Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Constraint Violation Counts"] = std::vector<size_t>({1});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Constraint Violation Counts"] = 1;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Constraint Evaluations"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Constraint Evaluations"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Normal Constraint Approximation"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Normal Constraint Approximation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Constraint Evaluations"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Constraint Evaluations"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Has Discrete Variables"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Has Discrete Variables"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Discrete Mutations"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Discrete Mutations"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Number Of Discrete Mutations"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Number Of Discrete Mutations"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Number Masking Matrix Entries"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Number Masking Matrix Entries"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Masking Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Masking Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Masking Matrix Sigma"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Masking Matrix Sigma"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Chi Square Number Discrete Mutations"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Chi Square Number Discrete Mutations"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Min Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Min Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Max Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Max Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Constraint Evaluation Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Constraint Evaluation Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Has Constraints"] = true;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Has Constraints"] = "Not a Boolean";
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mirrored Sampling"] = true;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Mirrored Sampling");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mirrored Sampling"] = "Not a Boolean";
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Population Size"] = 2;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Population Size");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Population Size"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Value"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Mu Value");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Type"] = "Linear";
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Type"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Type"] = "Undefined";
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));
  ASSERT_ANY_THROW(opt->initMuWeights(4));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Mu Type");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Sigma Cumulation Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Sigma Cumulation Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Initial Sigma Cumulation Factor");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Damp Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Damp Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Initial Damp Factor");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Use Gradient Information"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Use Gradient Information");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Use Gradient Information"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Step Size"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Gradient Step Size");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Step Size"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Sigma Bounded"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Is Sigma Bounded");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Sigma Bounded"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Cumulative Covariance"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Initial Cumulative Covariance");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Cumulative Covariance"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Diagonal Covariance"] = true;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Diagonal Covariance");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Diagonal Covariance"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Population Size"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Viability Population Size");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Population Size"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Mu Value"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Viability Mu Value");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Mu Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Covariance Matrix Corrections"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Max Covariance Matrix Corrections");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Covariance Matrix Corrections"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Target Success Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Target Success Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Target Success Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaption Strength"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Covariance Matrix Adaption Strength");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaption Strength"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Normal Vector Learning Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Normal Vector Learning Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Normal Vector Learning Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Global Success Learning Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Global Success Learning Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Global Success Learning Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Infeasible Resamplings"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Infeasible Resamplings");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Infeasible Resamplings"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Condition Covariance Matrix"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Condition Covariance Matrix");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Condition Covariance Matrix"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Standard Deviation");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Standard Deviation");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));
 
  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Granularity");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Granularity"] = "Not a Number";
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Granularity"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing changes in cumulative covariance
  opt->_initialCumulativeCovariance = -1.0;
  ASSERT_NO_THROW(opt->initMuWeights(4));
  ASSERT_NE(opt->_initialCumulativeCovariance, opt->_cumulativeCovariance);

  opt->_initialCumulativeCovariance = 0.5;
  ASSERT_NO_THROW(opt->initMuWeights(4));
  ASSERT_EQ(opt->_initialCumulativeCovariance, opt->_cumulativeCovariance);
 }

 //////////////// DEA ////////////////////////

 TEST(optimizers, DEA)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;

  // Creating optimizer configuration Json
  knlohmann::json optimizerJs;
  optimizerJs["Type"] = "Optimizer/DEA";

  // Creating module
  DEA* opt;
  ASSERT_NO_THROW(opt = dynamic_cast<DEA *>(Module::getModule(optimizerJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(opt->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = optimizerJs;
  auto baseExpJs = experimentJs;

  // Setting up optimizer correctly
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing infinite initial value fail
  v._lowerBound = 5.0;
  v._upperBound = -5.0;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());

  // Testing initial configuration success
  v._lowerBound = -5.0;
  v._upperBound = 5.0;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing optional parameters
  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Value Vector"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Value Vector"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Value Vector"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Value Vector"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Population"] = std::vector<std::vector<double>>({{ 2.0 }});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Population"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Candidate Population"] = std::vector<std::vector<double>>({{ 2.0 }});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Candidate Population"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Sample Index"] = 2;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Sample Index"] = std::vector<std::vector<double>>({{ 2.0 }});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Ever Value"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Ever Value"] = std::vector<std::vector<double>>({{ 2.0 }});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mean"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mean"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Mean"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Mean"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variables"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variables"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Distances"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Distances"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Infeasible Sample Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Infeasible Sample Count"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Minimum Step Size"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Minimum Step Size"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Population Size"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Population Size");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Population Size"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Crossover Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Crossover Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Crossover Rate"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mutation Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Mutation Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mutation Rate"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mutation Rule"] = "Fixed";
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Mutation Rule");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mutation Rule"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Selection Rule"] = "Best";
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Parent Selection Rule");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Selection Rule"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Accept Rule"] = "Best";
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Accept Rule");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Accept Rule"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Fix Infeasible"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Fix Infeasible");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Fix Infeasible"] = "Not a Number";
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Infeasible Resamplings"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Infeasible Resamplings");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Infeasible Resamplings"] = "Not a Number";
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Value"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Value");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Value"] = "Not a Number";
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Step Size"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Step Size");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Step Size"] = "Not a Number";
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));


  // Testing termination criteria
  e._currentGeneration = 2;
  opt->_infeasibleSampleCount  = 0;
  opt->_minValue  = 0.0;
  opt->_minStepSize  = 1.0;
  opt->_bestEverValue = -1.0;
  opt->_currentMinimumStepSize  = 2.0;

  opt->_bestEverValue = -1.0;
  ASSERT_FALSE(opt->checkTermination());
  opt->_bestEverValue = 1.0;
  ASSERT_TRUE(opt->checkTermination());
  opt->_bestEverValue= -1.0;

  opt->_currentMinimumStepSize = 2.0;
  ASSERT_FALSE(opt->checkTermination());
  opt->_currentMinimumStepSize = 0.1;
  ASSERT_TRUE(opt->checkTermination());
  opt->_currentMinimumStepSize= 2.0;
 }

 //////////////// MOCMAES ////////////////////////

 TEST(optimizers, MOCMAES)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;
  e["Problem"]["Type"] = "Optimization";

  // Creating problem module
  Optimization* p;
  knlohmann::json problemJs;
  problemJs["Type"] = "Optimization";
  ASSERT_NO_THROW(p = dynamic_cast<Optimization *>(Module::getModule(problemJs, &e)));
  e._problem = p;
  p->_numObjectives = 2;

  // Creating optimizer configuration Json
  knlohmann::json optimizerJs;
  optimizerJs["Type"] = "Optimizer/MOCMAES";

  // Creating module
  MOCMAES* opt;
  ASSERT_NO_THROW(opt = dynamic_cast<MOCMAES *>(Module::getModule(optimizerJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(opt->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = optimizerJs;
  auto baseExpJs = experimentJs;

  // Setting up optimizer correctly
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing initial configuration failure
  e._problem = NULL;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  e._problem = p;

  // Testing initial configuration failure
  p->_numObjectives = 1;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  p->_numObjectives = 2;

  // Testing initial configuration failure
  opt->_muValue = 64;
  opt->_populationSize = 32;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  opt->_muValue = 16;

  // Testing initial configuration failure
  opt->_successLearningRate = -1.0;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  opt->_successLearningRate = 0.5;

  // Testing initial configuration failure
  opt->_targetSuccessRate = -1.0;
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
  opt->_targetSuccessRate = 0.5;

  // Testing initial configuration success
  opt->_covarianceLearningRate = -1.0;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing initial configuration success
  opt->_evolutionPathAdaptionStrength = -1.0;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing initial configuration success
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing optional parameters
  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Non Dominated Sample Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Non Dominated Sample Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Values"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Values"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Values"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Values"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Index"] = std::vector<size_t>({1});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Index"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Sample Population"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Sample Population"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Sample Population"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Sample Population"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Sigma"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Sigma"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Sample Population"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Sample Population"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Sigma"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Sigma"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Sigma"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Sigma"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Num Objectives"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Num Objectives"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Covariance Matrix"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Covariance Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Covariance Matrix"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Covariance Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Covariance Matrix"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Covariance Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Evolution Paths"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Evolution Paths"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Evolution Paths"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Evolution Paths"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Evolution Paths"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Evolution Paths"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Success Probabilities"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Parent Success Probabilities"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Success Probabilities"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Success Probabilities"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Success Probabilities"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Success Probabilities"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Finished Sample Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Finished Sample Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Values"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Values"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Variables Vector"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Variables Vector"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Values"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Values"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Variables Vector"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Variables Vector"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Values"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Values"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variables Vector"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variables Vector"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Collection"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Collection"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Value Collection"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Value Collection"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Infeasible Sample Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Infeasible Sample Count"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Value Differences"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Value Differences"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variable Differences"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variable Differences"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Min Standard Deviations"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Min Standard Deviations"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Max Standard Deviations"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Max Standard Deviations"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Population Size"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Population Size");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Population Size"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Value"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Mu Value");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Evolution Path Adaption Strength"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Evolution Path Adaption Strength");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Evolution Path Adaption Strength"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Learning Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Covariance Learning Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Learning Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Target Success Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Target Success Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Target Success Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Threshold Probability"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Threshold Probability");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Threshold Probability"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Success Learning Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Success Learning Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Success Learning Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Max Value Difference Threshold"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Max Value Difference Threshold");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Max Value Difference Threshold"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Variable Difference Threshold"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Variable Difference Threshold");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Variable Difference Threshold"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Standard Deviation");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Standard Deviation");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  // Testing termination criteria
  e._currentGeneration = 2;
  opt->_maxGenerations = 10;
  opt->_minValueDifferenceThreshold = -std::numeric_limits<double>::infinity();
  opt->_minVariableDifferenceThreshold = -std::numeric_limits<double>::infinity();
  opt->_minStandardDeviation = -std::numeric_limits<double>::infinity();
  opt->_maxStandardDeviation = std::numeric_limits<double>::infinity();

  opt->_minValueDifferenceThreshold = 2.0;
  opt->_currentBestValueDifferences[0] = 3.0;
  ASSERT_FALSE(opt->checkTermination());
  opt->_currentBestValueDifferences[0] = 1.0;
  ASSERT_TRUE(opt->checkTermination());
  opt->_minValueDifferenceThreshold = -std::numeric_limits<double>::infinity();

  opt->_minVariableDifferenceThreshold = 2.0;
  opt->_currentBestVariableDifferences[0] = 3.0;
  ASSERT_FALSE(opt->checkTermination());
  opt->_currentBestVariableDifferences[0] = 1.0;
  ASSERT_TRUE(opt->checkTermination());
  opt->_minVariableDifferenceThreshold = -std::numeric_limits<double>::infinity();

  opt->_minStandardDeviation = 2.0;
  opt->_currentMinStandardDeviations[0] = 3.0;
  ASSERT_FALSE(opt->checkTermination());
  opt->_currentMinStandardDeviations[0] = 1.0;
  ASSERT_TRUE(opt->checkTermination());
  opt->_minStandardDeviation = -std::numeric_limits<double>::infinity();

  opt->_maxStandardDeviation = 2.0;
  opt->_currentMaxStandardDeviations[0] = 1.0;
  ASSERT_FALSE(opt->checkTermination());
  opt->_currentMaxStandardDeviations[0] = 3.0;
  ASSERT_TRUE(opt->checkTermination());
  opt->_maxStandardDeviation = std::numeric_limits<double>::infinity();
 }

 //////////////// MADGRAD ////////////////////////

  TEST(optimizers, MADGRAD)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating initial variable
   Variable v;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Lower Bound"] = 0.0;
   e["Variables"][0]["Upper Bound"] = 1.0;

   // Creating optimizer configuration Json
   knlohmann::json optimizerJs;
   optimizerJs["Type"] = "Optimizer/MADGRAD";

   // Creating module
   MADGRAD* opt;
   ASSERT_NO_THROW(opt = dynamic_cast<MADGRAD *>(Module::getModule(optimizerJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(opt->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = optimizerJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   // Testing infinite initial value fail
   v._initialValue = std::numeric_limits<double>::infinity();
   ASSERT_ANY_THROW(opt->setInitialConfiguration());

   // Testing initial configuration success
   v._initialValue = 1.0;
   ASSERT_NO_THROW(opt->setInitialConfiguration());

   // Testing optional parameters
   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Current Variable"] = std::vector<double>({ 2.0 });
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Current Variable"] = 2.0;
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Initial Parameter"] = std::vector<double>({ 2.0 });
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Initial Parameter"] = 2.0;
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Scaled Learning Rate"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Scaled Learning Rate"] = std::vector<double>({ 2.0 });
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Weight Decay"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs.erase("Weight Decay");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Weight Decay"] = std::vector<double>({ 2.0 });
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Gradient Sum"] = std::vector<double>({ 2.0 });
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Gradient Sum"] = 2.0;
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Squared Gradient Sum"] = std::vector<double>({ 2.0 });
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Squared Gradient Sum"] = 2.0;
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Gradient"] = std::vector<double>({ 2.0 });
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Gradient"] = 2.0;
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Best Ever Gradient"] = std::vector<double>({ 2.0 });
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Best Ever Gradient"] = 2.0;
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Gradient Norm"] = std::vector<double>({ 2.0 });
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Gradient Norm"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Eta"] = std::vector<double>({ 2.0 });
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs.erase("Eta");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Eta"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Epsilon"] = std::vector<double>({ 2.0 });
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs.erase("Epsilon");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Epsilon"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"]["Min Gradient Norm"] = std::vector<double>({ 1.0 });
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"].erase("Min Gradient Norm");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"]["Min Gradient Norm"] = 1.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"]["Max Gradient Norm"] = std::vector<double>({ 2.0 });
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"].erase("Max Gradient Norm");
   ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

   optimizerJs = baseOptJs;
   experimentJs = baseExpJs;
   optimizerJs["Termination Criteria"]["Max Gradient Norm"] = 2.0;
   ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

   // Testing termination criteria
   e._currentGeneration = 2;
   opt->_gradientNorm = 0.0;
   ASSERT_TRUE(opt->checkTermination());
   opt->_gradientNorm = 1.5;
   ASSERT_FALSE(opt->checkTermination());
   opt->_gradientNorm = 3.0;
   ASSERT_TRUE(opt->checkTermination());
  }

  //////////////// Rprop ////////////////////////

   TEST(optimizers, Rprop)
   {
    // Creating base experiment
    Experiment e;
    auto& experimentJs = e._js.getJson();

    // Creating initial variable
    Variable v;
    e._variables.push_back(&v);
    e["Variables"][0]["Name"] = "Var 1";
    e["Variables"][0]["Lower Bound"] = 0.0;
    e["Variables"][0]["Upper Bound"] = 1.0;

    // Creating optimizer configuration Json
    knlohmann::json optimizerJs;
    optimizerJs["Type"] = "Optimizer/Rprop";

    // Creating module
    Rprop* opt;
    ASSERT_NO_THROW(opt = dynamic_cast<Rprop *>(Module::getModule(optimizerJs, &e)));

    // Defaults should be applied without a problem
    ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

    // Covering variable functions (no effect)
    ASSERT_NO_THROW(opt->applyVariableDefaults());

    // Backup the correct base configuration
    auto baseOptJs = optimizerJs;
    auto baseExpJs = experimentJs;

    // Setting up optimizer correctly
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    // Testing infinite initial value fail
    v._initialValue = std::numeric_limits<double>::infinity();
    ASSERT_ANY_THROW(opt->setInitialConfiguration());

    // Testing initial configuration success
    v._initialValue = 1.0;
    ASSERT_NO_THROW(opt->setInitialConfiguration());

    // Testing optional parameters
    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Current Variable"] = std::vector<double>({ 2.0 });
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Current Variable"] = 2.0;
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Best Ever Variable"] = std::vector<double>({ 2.0 });
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Best Ever Variable"] = 2.0;
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Delta"] = std::vector<double>({ 2.0 });
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Delta"] = 2.0;
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Current Gradient"] = std::vector<double>({ 2.0 });
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Current Gradient"] = 2.0;
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Previous Gradient"] = std::vector<double>({ 2.0 });
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Previous Gradient"] = 2.0;
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Best Ever Gradient"] = std::vector<double>({ 2.0 });
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Best Ever Gradient"] = 2.0;
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Norm Previous Gradient"] = 2.0;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Norm Previous Gradient"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["X Diff"] = 2.0;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["X Diff"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Delta0"] = 2.0;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs.erase("Delta0");
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Delta0"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Delta Min"] = 2.0;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Delta Min"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs.erase("Delta Min");
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Delta Max"] = 2.0;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs.erase("Delta Max");
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Delta Max"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Eta Minus"] = 2.0;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs.erase("Eta Minus");
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Eta Minus"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Eta Plus"] = 2.0;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs.erase("Eta Plus");
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Eta Plus"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Max Stall Counter"] = 2;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Max Stall Counter"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Termination Criteria"]["Max Stall Generations"] = std::vector<double>({ 1.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Termination Criteria"].erase("Max Stall Generations");
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Termination Criteria"]["Max Stall Generations"] = 1;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Termination Criteria"]["Parameter Relative Tolerance"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Termination Criteria"].erase("Parameter Relative Tolerance");
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Termination Criteria"]["Parameter Relative Tolerance"] = 2.0;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Termination Criteria"]["Max Gradient Norm"] = std::vector<double>({ 2.0 });
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Termination Criteria"].erase("Max Gradient Norm");
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Termination Criteria"]["Max Gradient Norm"] = 2.0;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    // Testing termination criteria
    e._currentGeneration = 200;
    opt->_maxStallGenerations = 100;
    opt->_maxGradientNorm = 1.0;
    opt->_parameterRelativeTolerance = 1.5;

    opt->_maxStallCounter = 0;
    opt->_normPreviousGradient = 2.0;
    opt->_xDiff = 2.0;

    opt->_maxStallCounter = 200;
    ASSERT_TRUE(opt->checkTermination());
    opt->_maxStallCounter = 50;
    ASSERT_FALSE(opt->checkTermination());

    opt->_normPreviousGradient = 0.0;
    ASSERT_TRUE(opt->checkTermination());
    opt->_normPreviousGradient = 2.0;
    ASSERT_FALSE(opt->checkTermination());

    opt->_xDiff = 0.5;
    ASSERT_TRUE(opt->checkTermination());
    opt->_xDiff = 1.5;
    ASSERT_FALSE(opt->checkTermination());
   }

   //////////////// GridSearch ////////////////////////

   TEST(optimizers, GridSearch)
   {
    // Creating base experiment
    Experiment e;
    auto& experimentJs = e._js.getJson();

    // Creating initial variable
    Variable v;
    e._variables.push_back(&v);
    e["Variables"][0]["Name"] = "Var 1";

    // Creating optimizer configuration Json
    knlohmann::json optimizerJs;
    optimizerJs["Type"] = "Optimizer/GridSearch";

    // Creating module
    GridSearch* opt;
    ASSERT_NO_THROW(opt = dynamic_cast<GridSearch *>(Module::getModule(optimizerJs, &e)));

    // Defaults should be applied without a problem
    ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

    // Covering variable functions (no effect)
    ASSERT_NO_THROW(opt->applyVariableDefaults());

    // Backup the correct base configuration
    auto baseOptJs = optimizerJs;
    auto baseExpJs = experimentJs;

    // Testing initial configuration fail
    opt->_numberOfValues = 10;
    opt->_maxModelEvaluations = 5;
    ASSERT_NO_THROW(opt->setInitialConfiguration());
    opt->_maxModelEvaluations = 20;

    // Testing initial configuration success
    ASSERT_NO_THROW(opt->setInitialConfiguration());

    // Testing optional parameters
    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Number Of Values"] = 1;
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Number Of Values"] = "Not a Number";
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Objective"] = std::vector<double>({ 2.0 });
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Objective"] = 2.0;
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Index Helper"] = std::vector<size_t>({ 0 });
    ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

    optimizerJs = baseOptJs;
    experimentJs = baseExpJs;
    optimizerJs["Index Helper"] = 2.0;
    ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));
   }

} // namespace
