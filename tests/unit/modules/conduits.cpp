#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/conduit/distributed/distributed.hpp"
#include "modules/conduit/concurrent/concurrent.hpp"
#include "modules/conduit/sequential/sequential.hpp"

namespace korali { namespace conduit {
extern void _workerWrapper();
extern Sequential *_currentConduit;
}}

namespace
{
 using namespace korali;
 using namespace korali::conduit;

 TEST(Conduit, SequentialConduit)
 {

  knlohmann::json moduleJs;
  ASSERT_ANY_THROW(Module::getModule(moduleJs, NULL));
  moduleJs["Type"] = 1.0;
  ASSERT_ANY_THROW(Module::getModule(moduleJs, NULL));
  moduleJs["Type"] = "Sequential";
  Module* m;
  ASSERT_NO_THROW(m = Module::getModule(moduleJs, NULL));
  ASSERT_NO_THROW(m->initialize());
  ASSERT_NO_THROW(m->finalize());
  ASSERT_NO_THROW(m->getType());
  ASSERT_NO_THROW(m->checkTermination());
  ASSERT_NO_THROW(m->getConfiguration(moduleJs));
  ASSERT_NO_THROW(m->setConfiguration(moduleJs));
  ASSERT_NO_THROW(m->applyModuleDefaults(moduleJs));
  ASSERT_NO_THROW(m->applyVariableDefaults());
  Sample s;
  ASSERT_NO_THROW(m->runOperation("A", s));


  knlohmann::json conduitJs;
  conduitJs["Type"] = "Sequential";

  // Creating module
  Sequential* conduit;
  ASSERT_NO_THROW(conduit = dynamic_cast<Sequential *>(Module::getModule(conduitJs, NULL)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(conduit->applyModuleDefaults(conduitJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(conduit->applyVariableDefaults());

  // Setting up conduit
  ASSERT_NO_THROW(conduit->setConfiguration(conduitJs));

  // Initializing server
  ASSERT_NO_THROW(conduit->initServer());
  ASSERT_NO_THROW(conduit->getProcessId());

  // Broadcasting message
  knlohmann::json message;
  message["Conduit Action"] == "Terminate";
  ASSERT_NO_THROW(conduit->broadcastMessageToWorkers(message));

  _currentConduit = NULL;
  _workerWrapper();
 }

 TEST(Conduit, ConcurrentConduit)
 {
  knlohmann::json conduitJs;
  conduitJs["Type"] = "Concurrent";

  // Creating module
  Concurrent* conduit;
  ASSERT_NO_THROW(conduit = dynamic_cast<Concurrent *>(Module::getModule(conduitJs, NULL)));

  // Testing configuration without mandatory field(s)
  ASSERT_ANY_THROW(conduit->setConfiguration(conduitJs));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(conduit->applyModuleDefaults(conduitJs));

  // Testing wrong value type (string) for the concurrent jobs parameter
  conduitJs["Concurrent Jobs"] = "16";
  ASSERT_ANY_THROW(conduit->setConfiguration(conduitJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(conduit->applyVariableDefaults());

  // Testing correct configuration value type
  conduitJs["Concurrent Jobs"] = 16;
  ASSERT_NO_THROW(conduit->setConfiguration(conduitJs));
 }

 TEST(Conduit, DistributedConduit)
 {
  knlohmann::json conduitJs;
  conduitJs["Type"] = "Distributed";

  // Creating module
  Distributed* conduit;
  ASSERT_NO_THROW(conduit = dynamic_cast<Distributed *>(Module::getModule(conduitJs, NULL)));

  // Testing configuration without mandatory field(s)
  ASSERT_ANY_THROW(conduit->setConfiguration(conduitJs));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(conduit->applyModuleDefaults(conduitJs));

  // Testing wrong value type (string) for the concurrent jobs parameter
  conduitJs["Ranks Per Worker"] = "16";
  ASSERT_ANY_THROW(conduit->setConfiguration(conduitJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(conduit->applyVariableDefaults());

  // Testing correct configuration value type
  conduitJs["Ranks Per Worker"] = 16;
  ASSERT_NO_THROW(conduit->setConfiguration(conduitJs));
 }

} // namespace
