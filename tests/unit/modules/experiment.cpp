#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/experiment/experiment.hpp"
#include "sample/sample.hpp"

namespace
{
 using namespace korali;

 TEST(Experiment, cornerCases)
 {
  Engine k;
  knlohmann::json expJs;
  expJs["Type"] = "Experiment";
  expJs["Problem"]["Type"] = "Optimization";
  expJs["Problem"]["Objective Function"] = 0;
  expJs["Solver"]["Type"] = "Optimizer/CMAES";
  expJs["Solver"]["Population Size"] = 16;
  expJs["Solver"]["Termination Criteria"]["Max Generations"] = 1;
  expJs["Variables"][0]["Name"] = "Var 1";
  expJs["Variables"][0]["Lower Bound"] = -1.0;
  expJs["Variables"][0]["Upper Bound"] = 1.0;

  // Creating module
  Experiment* e;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  ASSERT_NO_THROW(e->applyModuleDefaults(expJs));
  ASSERT_NO_THROW(e->applyVariableDefaults());

  expJs["Type"] = "Experiment";
  auto backJs = expJs;
  e->_js.getJson() = expJs;

  e->_experimentId = 0;
  e->_engine = &k;
  e->_isFinished = false;
  ASSERT_NO_THROW(e->initialize());

  expJs = backJs;
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Current Generation"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Current Generation"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Is Finished"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Is Finished"] = true;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Run ID"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Run ID"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Timestamp"] = 1.0;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Timestamp"] = "00.00.00";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Random Seed");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Random Seed"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));
  expJs["Random Seed"] = "Not a Number";
  ASSERT_ANY_THROW(e->setSeed(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Random Seed"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));
  expJs["Random Seed"] = 1;
  ASSERT_NO_THROW(e->setSeed(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Distributions");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Problem");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Solver");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Preserve Random Number Generator States");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Preserve Random Number Generator States"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Preserve Random Number Generator States"] = true;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"].erase("Path");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Path"] = 1.0;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Path"] = "Path";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"].erase("Enabled");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"].erase("Use Multiple Files");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Use Multiple Files"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Use Multiple Files"] = true;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Enabled"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Enabled"] = true;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"].erase("Frequency");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Frequency"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["File Output"]["Frequency"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs.erase("Store Sample Information");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Store Sample Information"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Store Sample Information"] = false;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"].erase("Verbosity");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"]["Verbosity"] = 1.0;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"]["Verbosity"] = "Silent";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"].erase("Frequency");
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"]["Frequency"] = "Not a Number";
  e->initialize();
  expJs.erase("Variables");
  ASSERT_ANY_THROW(e->setConfiguration(expJs));

  expJs = backJs;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  expJs["Console Output"]["Frequency"] = 1;
  e->initialize();
  expJs.erase("Variables");
  ASSERT_NO_THROW(e->setConfiguration(expJs));
 }

 TEST(Experiment, execution)
 {
  Engine k;
  knlohmann::json expJs;
  expJs["Type"] = "Experiment";
  expJs["Problem"]["Type"] = "Optimization";
  expJs["Problem"]["Objective Function"] = 0;
  expJs["Solver"]["Type"] = "Optimizer/CMAES";
  expJs["Solver"]["Population Size"] = 16;
  expJs["Solver"]["Termination Criteria"]["Max Generations"] = 1;
  expJs["Variables"][0]["Name"] = "Var 1";
  expJs["Variables"][0]["Lower Bound"] = -1.0;
  expJs["Variables"][0]["Upper Bound"] = 1.0;

  // Creating module
  Experiment* e;
  ASSERT_NO_THROW(e = dynamic_cast<Experiment *>(Module::getModule(expJs, NULL)));
  ASSERT_NO_THROW(e->applyModuleDefaults(expJs));
  ASSERT_NO_THROW(e->applyVariableDefaults());

  expJs["Type"] = "Experiment";
  auto backJs = expJs;
  e->_js.getJson() = expJs;

  e->_experimentId = 0;
  e->_engine = &k;
  e->_isFinished = false;
  ASSERT_NO_THROW(e->initialize());

  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["F(x)"] = 0.1;
  };
  _functionVector.push_back(&modelFc);
  ASSERT_NO_THROW(k.run(*e));

  ASSERT_NO_THROW(delete e);
 }

} // namespace
