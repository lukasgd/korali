#include "gtest/gtest.h"
#include "modules/neuralNetwork/neuralNetwork.hpp"
#include "modules/neuralNetwork/layer/layer.hpp"
#include "modules/neuralNetwork/layer/activation/activation.hpp"
#include "modules/neuralNetwork/layer/input/input.hpp"
#include "modules/neuralNetwork/layer/linear/linear.hpp"
#include "modules/neuralNetwork/layer/output/output.hpp"
#include "modules/neuralNetwork/layer/recurrent/recurrent.hpp"
#include "modules/neuralNetwork/layer/recurrent/gru/gru.hpp"
#include "modules/neuralNetwork/layer/recurrent/lstm/lstm.hpp"
#include "korali.hpp"

namespace
{
 using namespace korali;
 using namespace korali::neuralNetwork;
 using namespace korali::neuralNetwork::layer;
 using namespace korali::neuralNetwork::layer::recurrent;

 /****************************************************************************
  * OneDNN
  ***************************************************************************/

 TEST(NeuralNetwork, BaseConfigOneDNN)
 {
  Experiment e;
  e._logger = new Logger("Detailed", stdout);

  NeuralNetwork* nn;
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = "OneDNN";
  neuralNetworkConfig["Timestep Count"] = 1;
  neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
  neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Output";
  neuralNetworkConfig["Mode"] = "Training";

  ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
  ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
  auto baseNNJs = neuralNetworkConfig;

  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
  ASSERT_NO_THROW(nn->applyVariableDefaults());
  ASSERT_NO_THROW(nn->initialize());
  ASSERT_ANY_THROW(nn->initialize());

  std::vector<float> _hyperparameters;
  ASSERT_NO_THROW(_hyperparameters = nn->generateInitialHyperparameters());
  ASSERT_NO_THROW(nn->setHyperparameters(_hyperparameters));
  ASSERT_ANY_THROW(nn->setHyperparameters(std::vector<float>({0.1, 0.1})));

  nn->_isInitialized = false;
  nn->_engine = "CuDNN";
  ASSERT_NO_THROW(nn->initialize());

  nn->_isInitialized = false;
  nn->_batchSizes = std::vector<size_t>();
  ASSERT_ANY_THROW(nn->initialize());

  nn->_isInitialized = false;
  nn->_batchSizes = std::vector<size_t>({0});
  ASSERT_ANY_THROW(nn->initialize());

  nn->_batchSizes = std::vector<size_t>({1});
  nn->_isInitialized = false;
  ASSERT_NO_THROW(nn->initialize());

  ASSERT_ANY_THROW(nn->forward({{{}}, {{}}}));
  ASSERT_ANY_THROW(nn->forward({{{}, {}}}));
  ASSERT_ANY_THROW(nn->forward({{{}}}));
  ASSERT_NO_THROW(nn->forward({{{0.0}}}));
  ASSERT_NO_THROW(nn->backward({{0.0}}));
  ASSERT_ANY_THROW(nn->backward({{0.0, 0.0}}));

  nn->_mode = "Inference";
  ASSERT_ANY_THROW(nn->backward({{0.0}}));

  ASSERT_NO_THROW(nn->getInputGradients(1));
  ASSERT_ANY_THROW(nn->getInputGradients(0));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Current Training Loss"] = "Not a Number";
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Current Training Loss"] = 1.0;
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Engine");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Engine"] = 1.0;
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Engine"] = "Korali";
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Mode");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Mode"] = 1.0;
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Mode"] = "Training";
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Layers");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Layers"] = knlohmann::json();
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Timestep Count");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Timestep Count"] = "Not a Number";
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Timestep Count"] = 1;
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig.erase("Batch Sizes");
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Batch Sizes"] = "Not a Number";
  ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

  neuralNetworkConfig = baseNNJs;
  neuralNetworkConfig["Batch Sizes"] = std::vector<float>({1});
  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

  ASSERT_NO_THROW(nn->getConfiguration(neuralNetworkConfig));
 }

 TEST(NeuralNetwork, ActivationLayerOneDNN)
  {
   Experiment e;
   e._logger = new Logger("Detailed", stdout);

   NeuralNetwork* nn;
   knlohmann::json neuralNetworkConfig;
   neuralNetworkConfig["Type"] = "Neural Network";
   neuralNetworkConfig["Engine"] = "OneDNN";
   neuralNetworkConfig["Timestep Count"] = 1;
   neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
   neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
   neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Activation";
   neuralNetworkConfig["Layers"][1]["Function"] = "Elementwise/Linear";
   neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
   neuralNetworkConfig["Mode"] = "Training";

   ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
   ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
   auto baseNNJs = neuralNetworkConfig;

   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
   ASSERT_NO_THROW(nn->applyVariableDefaults());
   ASSERT_NO_THROW(nn->initialize());

   Activation* layer = dynamic_cast<Activation*>(nn->_pipelines[0][0]._layerVector[1]);
   ASSERT_NO_THROW(layer->applyVariableDefaults());
   knlohmann::json layerJs;
   ASSERT_NO_THROW(layer->getConfiguration(layerJs));

   nn->_mode = "Inference";
   ASSERT_ANY_THROW(layer->backwardData(0));
   ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
   ASSERT_ANY_THROW(layer->createBackwardPipeline());
   nn->_mode = "Training";
   ASSERT_NO_THROW(layer->backwardData(0));
   ASSERT_NO_THROW(layer->backwardHyperparameters(0));
   ASSERT_NO_THROW(layer->createBackwardPipeline());

   layer->_function = "Elementwise/Clip";
   layer->_alpha = 0.3;
   layer->_beta = 0.6;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.5, 0.000001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/Clip";
   layer->_alpha = 0.3;
   layer->_beta = 0.4;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.4, 0.000001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/Clip";
   layer->_alpha = 0.9;
   layer->_beta = 1.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.9, 0.000001);
   ASSERT_NO_THROW(nn->backward({{{0.9}}}));

   layer->_function = "Elementwise/Log";
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], -0.69314718, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/Linear";
   layer->_alpha = 2.0;
   layer->_beta = 0.6;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 1.6, 0.000001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/ReLU";
   layer->_alpha = 0.1;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.5, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/ReLU";
   layer->_alpha = 0.1;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{-0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], -0.05, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/Tanh";
   layer->_alpha = 0.0;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.46211715, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));
 
   layer->_function = "Elementwise/SoftReLU";
   layer->_alpha = 0.0;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.9740769842, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/Logistic";
   layer->_alpha = 0.0;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.62245933120, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   layer->_function = "Elementwise/SoftSign";
   layer->_alpha = 0.0;
   layer->_beta = 0.0;
   ASSERT_ANY_THROW(layer->createForwardPipeline());

   layer->_function = "Softmax";
   layer->_alpha = 0.0;
   layer->_beta = 0.0;
   ASSERT_NO_THROW(layer->createForwardPipeline());
   ASSERT_NO_THROW(nn->forward({{{0.5}}}));
   ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 1.0, 0.001);
   ASSERT_NO_THROW(nn->backward({{{0.5}}}));

   knlohmann::json baseLayerJs = layerJs;

   layerJs = baseLayerJs;
   layerJs.erase("Function");
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Function"] = 1.0;
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Function"] = "Unknown";
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Function"] = "Softmax";
   ASSERT_NO_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs.erase("Alpha");
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Alpha"] = "Not a Number";
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Alpha"] = 1.0;
   ASSERT_NO_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs.erase("Beta");
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Beta"] = "Not a Number";
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Beta"] = 1.0;
   ASSERT_NO_THROW(layer->setConfiguration(layerJs));

   Layer* baseLayer = new Activation;
   delete baseLayer;
  }

 TEST(NeuralNetwork, InputLayerOneDNN)
 {
  Experiment e;
  e._logger = new Logger("Detailed", stdout);

  NeuralNetwork* nn;
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = "OneDNN";
  neuralNetworkConfig["Timestep Count"] = 1;
  neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
  neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Activation";
  neuralNetworkConfig["Layers"][1]["Function"] = "Elementwise/Linear";
  neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
  neuralNetworkConfig["Mode"] = "Training";

  ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
  ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
  auto baseNNJs = neuralNetworkConfig;

  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
  ASSERT_NO_THROW(nn->applyVariableDefaults());
  ASSERT_NO_THROW(nn->initialize());

  Input* layer = dynamic_cast<Input*>(nn->_pipelines[0][0]._layerVector[0]);
  ASSERT_NO_THROW(layer->applyVariableDefaults());
  knlohmann::json layerJs;
  ASSERT_NO_THROW(layer->getConfiguration(layerJs));

  nn->_mode = "Inference";
  ASSERT_ANY_THROW(layer->backwardData(0));
  ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
  ASSERT_ANY_THROW(layer->createBackwardPipeline());
  nn->_mode = "Training";
  ASSERT_NO_THROW(layer->backwardData(0));
  ASSERT_NO_THROW(layer->backwardHyperparameters(0));
  ASSERT_NO_THROW(layer->createBackwardPipeline());
 }

 TEST(NeuralNetwork, OutputLayerOneDNN)
 {
  Experiment e;
  e._logger = new Logger("Detailed", stdout);

  NeuralNetwork* nn;
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = "OneDNN";
  neuralNetworkConfig["Timestep Count"] = 1;
  neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
  neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Output";
  neuralNetworkConfig["Mode"] = "Training";

  ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
  ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
  auto baseNNJs = neuralNetworkConfig;

  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
  ASSERT_NO_THROW(nn->applyVariableDefaults());
  ASSERT_NO_THROW(nn->initialize());

  Output* layer = dynamic_cast<Output*>(nn->_pipelines[0][0]._layerVector[1]);
  ASSERT_NO_THROW(layer->applyVariableDefaults());
  knlohmann::json layerJs;
  ASSERT_NO_THROW(layer->getConfiguration(layerJs));

  nn->_mode = "Inference";
  ASSERT_ANY_THROW(layer->backwardData(0));
  ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
  ASSERT_ANY_THROW(layer->createBackwardPipeline());
  nn->_mode = "Training";
  ASSERT_NO_THROW(layer->backwardData(0));
  ASSERT_NO_THROW(layer->backwardHyperparameters(0));
  ASSERT_NO_THROW(layer->getOutput());

  ASSERT_NO_THROW(layer->initialize());

  layer->_index = 0;
  ASSERT_ANY_THROW(layer->initialize());
  layer->_index = nn->_layers.size()-1;
  ASSERT_NO_THROW(layer->initialize());

  layer->_scale = std::vector<float>({1.0, 1.0});
  ASSERT_ANY_THROW(layer->initialize());
  layer->_scale = std::vector<float>({1.0});
  ASSERT_NO_THROW(layer->initialize());

  layer->_shift = std::vector<float>({1.0, 1.0});
  ASSERT_ANY_THROW(layer->initialize());
  layer->_shift = std::vector<float>({1.0});
  ASSERT_NO_THROW(layer->initialize());

  layer->_transformationMask = std::vector<std::string>({"Identity", "Identity"});
  ASSERT_ANY_THROW(layer->initialize());
  layer->_transformationMask = std::vector<std::string>({"Unknown"});
  ASSERT_ANY_THROW(layer->initialize());
  layer->_transformationMask = std::vector<std::string>({"Identity"});
  ASSERT_NO_THROW(layer->initialize());
  layer->_transformationMask = std::vector<std::string>({"Absolute"});
  ASSERT_NO_THROW(layer->initialize());
  layer->_transformationMask = std::vector<std::string>({"Softplus"});
  ASSERT_NO_THROW(layer->initialize());
  layer->_transformationMask = std::vector<std::string>({"Tanh"});
  ASSERT_NO_THROW(layer->initialize());
  ASSERT_NO_THROW(layer->initialize());
  layer->_transformationMask = std::vector<std::string>({"Sigmoid"});
  ASSERT_NO_THROW(layer->initialize());

  knlohmann::json baseLayerJs = layerJs;

  layerJs = baseLayerJs;
  layerJs.erase("Output Channels");
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Output Channels"] = "Not a Number";
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Output Channels"] = 1;
  ASSERT_NO_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs.erase("Weight Scaling");
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Weight Scaling"] = "Not a Number";
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Weight Scaling"] = 1.0;
  ASSERT_NO_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs.erase("Transformation Mask");
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Transformation Mask"] = 1.0;
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Transformation Mask"] = std::vector<std::string>({"Uniform"});
  ASSERT_NO_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs.erase("Shift");
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Shift"] = "Not a Number";
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Shift"] = std::vector<float>(1.0);
  ASSERT_NO_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs.erase("Scale");
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Scale"] = "Not a Number";
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Scale"] = std::vector<float>(1.0);
  ASSERT_NO_THROW(layer->setConfiguration(layerJs));
 }

 TEST(NeuralNetwork, LinearLayerOneDNN)
 {
  Experiment e;
  e._logger = new Logger("Detailed", stdout);

  NeuralNetwork* nn;
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = "OneDNN";
  neuralNetworkConfig["Timestep Count"] = 1;
  neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
  neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Linear";
  neuralNetworkConfig["Layers"][1]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
  neuralNetworkConfig["Mode"] = "Training";

  ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
  ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
  auto baseNNJs = neuralNetworkConfig;

  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
  ASSERT_NO_THROW(nn->applyVariableDefaults());
  ASSERT_NO_THROW(nn->initialize());

  Linear* layer = dynamic_cast<Linear*>(nn->_pipelines[0][0]._layerVector[1]);
  ASSERT_NO_THROW(layer->applyVariableDefaults());
  knlohmann::json layerJs;
  ASSERT_NO_THROW(layer->getConfiguration(layerJs));

  nn->_mode = "Inference";
  ASSERT_ANY_THROW(layer->backwardData(0));
  ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
  ASSERT_ANY_THROW(layer->createBackwardPipeline());
  nn->_mode = "Training";
  ASSERT_NO_THROW(layer->backwardData(0));
  ASSERT_NO_THROW(layer->backwardHyperparameters(0));
  ASSERT_NO_THROW(layer->createBackwardPipeline());
 }

 TEST(NeuralNetwork, GRULayerOneDNN)
 {
  Experiment e;
  e._logger = new Logger("Detailed", stdout);

  NeuralNetwork* nn;
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = "OneDNN";
  neuralNetworkConfig["Timestep Count"] = 1;
  neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
  neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Recurrent/GRU";
  neuralNetworkConfig["Layers"][1]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
  neuralNetworkConfig["Mode"] = "Training";

  ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
  ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
  auto baseNNJs = neuralNetworkConfig;

  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
  ASSERT_NO_THROW(nn->applyVariableDefaults());
  ASSERT_NO_THROW(nn->initialize());

  GRU* layer = dynamic_cast<GRU*>(nn->_pipelines[0][0]._layerVector[1]);
  ASSERT_NO_THROW(layer->applyVariableDefaults());
  knlohmann::json layerJs;
  ASSERT_NO_THROW(layer->getConfiguration(layerJs));

  nn->_mode = "Inference";
  ASSERT_ANY_THROW(layer->backwardData(0));
  ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
  ASSERT_ANY_THROW(layer->createBackwardPipeline());
  nn->_mode = "Training";
  ASSERT_NO_THROW(layer->backwardData(0));
  ASSERT_NO_THROW(layer->backwardHyperparameters(0));
  ASSERT_NO_THROW(layer->createBackwardPipeline());
 }

 TEST(NeuralNetwork, LSTMLayerOneDNN)
 {
  Experiment e;
  e._logger = new Logger("Detailed", stdout);

  NeuralNetwork* nn;
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = "OneDNN";
  neuralNetworkConfig["Timestep Count"] = 1;
  neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
  neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Recurrent/LSTM";
  neuralNetworkConfig["Layers"][1]["Output Channels"] = 1;
  neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
  neuralNetworkConfig["Mode"] = "Training";

  ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
  ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
  auto baseNNJs = neuralNetworkConfig;

  ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
  ASSERT_NO_THROW(nn->applyVariableDefaults());
  ASSERT_NO_THROW(nn->initialize());

  LSTM* layer = dynamic_cast<LSTM*>(nn->_pipelines[0][0]._layerVector[1]);
  ASSERT_NO_THROW(layer->applyVariableDefaults());
  knlohmann::json layerJs;
  ASSERT_NO_THROW(layer->getConfiguration(layerJs));

  ASSERT_NO_THROW(layer->initialize());
  layer->_depth = 10;
  layer->_outputChannels = 10;
  ASSERT_ANY_THROW(layer->initialize());
  layer->_outputChannels = 1;
  ASSERT_NO_THROW(layer->initialize());
  layer->_depth = 1;
  ASSERT_NO_THROW(layer->initialize());

  nn->_mode = "Inference";
  ASSERT_ANY_THROW(layer->backwardData(0));
  ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
  ASSERT_ANY_THROW(layer->createBackwardPipeline());
  nn->_mode = "Training";
  ASSERT_NO_THROW(layer->backwardData(0));
  ASSERT_NO_THROW(layer->backwardHyperparameters(0));
  ASSERT_NO_THROW(layer->createBackwardPipeline());

  knlohmann::json baseLayerJs = layerJs;

  layerJs = baseLayerJs;
  layerJs.erase("Depth");
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Depth"] = "Not a Number";
  ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

  layerJs = baseLayerJs;
  layerJs["Depth"] = 1;
  ASSERT_NO_THROW(layer->setConfiguration(layerJs));
 }

 /****************************************************************************
  * Korali
  ***************************************************************************/

  TEST(NeuralNetwork, BaseConfigKorali)
  {
   Experiment e;
   e._logger = new Logger("Detailed", stdout);

   NeuralNetwork* nn;
   knlohmann::json neuralNetworkConfig;
   neuralNetworkConfig["Type"] = "Neural Network";
   neuralNetworkConfig["Engine"] = "Korali";
   neuralNetworkConfig["Timestep Count"] = 1;
   neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
   neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
   neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Output";
   neuralNetworkConfig["Mode"] = "Training";

   ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
   ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
   auto baseNNJs = neuralNetworkConfig;

   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
   ASSERT_NO_THROW(nn->applyVariableDefaults());
   ASSERT_NO_THROW(nn->initialize());
   ASSERT_ANY_THROW(nn->initialize());

   std::vector<float> _hyperparameters;
   ASSERT_NO_THROW(_hyperparameters = nn->generateInitialHyperparameters());
   ASSERT_NO_THROW(nn->setHyperparameters(_hyperparameters));
   ASSERT_ANY_THROW(nn->setHyperparameters(std::vector<float>({0.1, 0.1})));

   nn->_isInitialized = false;
   nn->_engine = "CuDNN";
   ASSERT_NO_THROW(nn->initialize());

   nn->_isInitialized = false;
   nn->_batchSizes = std::vector<size_t>();
   ASSERT_ANY_THROW(nn->initialize());

   nn->_isInitialized = false;
   nn->_batchSizes = std::vector<size_t>({0});
   ASSERT_ANY_THROW(nn->initialize());

   nn->_batchSizes = std::vector<size_t>({1});
   nn->_isInitialized = false;
   ASSERT_NO_THROW(nn->initialize());

   ASSERT_ANY_THROW(nn->forward({{{}}, {{}}}));
   ASSERT_ANY_THROW(nn->forward({{{}, {}}}));
   ASSERT_ANY_THROW(nn->forward({{{}}}));
   ASSERT_NO_THROW(nn->forward({{{0.0}}}));
   ASSERT_NO_THROW(nn->backward({{0.0}}));
   ASSERT_ANY_THROW(nn->backward({{0.0, 0.0}}));

   nn->_mode = "Inference";
   ASSERT_ANY_THROW(nn->backward({{0.0}}));

   ASSERT_NO_THROW(nn->getInputGradients(1));
   ASSERT_ANY_THROW(nn->getInputGradients(0));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Current Training Loss"] = "Not a Number";
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Current Training Loss"] = 1.0;
   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig.erase("Engine");
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Engine"] = 1.0;
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Engine"] = "Korali";
   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig.erase("Mode");
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Mode"] = 1.0;
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Mode"] = "Training";
   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig.erase("Layers");
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Layers"] = knlohmann::json();
   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig.erase("Timestep Count");
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Timestep Count"] = "Not a Number";
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Timestep Count"] = 1;
   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig.erase("Batch Sizes");
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Batch Sizes"] = "Not a Number";
   ASSERT_ANY_THROW(nn->setConfiguration(neuralNetworkConfig));

   neuralNetworkConfig = baseNNJs;
   neuralNetworkConfig["Batch Sizes"] = std::vector<float>({1});
   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));

   ASSERT_NO_THROW(nn->getConfiguration(neuralNetworkConfig));
  }

  TEST(NeuralNetwork, ActivationLayerKorali)
   {
    Experiment e;
    e._logger = new Logger("Detailed", stdout);

    NeuralNetwork* nn;
    knlohmann::json neuralNetworkConfig;
    neuralNetworkConfig["Type"] = "Neural Network";
    neuralNetworkConfig["Engine"] = "Korali";
    neuralNetworkConfig["Timestep Count"] = 1;
    neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
    neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
    neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
    neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Activation";
    neuralNetworkConfig["Layers"][1]["Function"] = "Elementwise/Linear";
    neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
    neuralNetworkConfig["Mode"] = "Training";

    ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
    ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
    auto baseNNJs = neuralNetworkConfig;

    ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
    ASSERT_NO_THROW(nn->applyVariableDefaults());
    ASSERT_NO_THROW(nn->initialize());

    Activation* layer = dynamic_cast<Activation*>(nn->_pipelines[0][0]._layerVector[1]);
    ASSERT_NO_THROW(layer->applyVariableDefaults());
    knlohmann::json layerJs;
    ASSERT_NO_THROW(layer->getConfiguration(layerJs));

    nn->_mode = "Inference";
    ASSERT_ANY_THROW(layer->backwardData(0));
    ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
    ASSERT_ANY_THROW(layer->createBackwardPipeline());
    nn->_mode = "Training";
    ASSERT_NO_THROW(layer->backwardData(0));
    ASSERT_NO_THROW(layer->backwardHyperparameters(0));
    ASSERT_NO_THROW(layer->createBackwardPipeline());

    layer->_function = "Elementwise/Clip";
    layer->_alpha = 0.3;
    layer->_beta = 0.6;
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.5, 0.000001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    layer->_function = "Elementwise/Clip";
    layer->_alpha = 0.3;
    layer->_beta = 0.4;
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.4, 0.000001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    layer->_function = "Elementwise/Log";
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], -0.693147180, 0.001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    layer->_function = "Elementwise/Linear";
    layer->_alpha = 2.0;
    layer->_beta = 0.6;
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 1.6, 0.000001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    layer->_function = "Elementwise/ReLU";
    layer->_alpha = 0.1;
    layer->_beta = 0.0;
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.5, 0.001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    layer->_function = "Elementwise/ReLU";
    layer->_alpha = 0.1;
    layer->_beta = 0.0;
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{-0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], -0.05, 0.001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    layer->_function = "Elementwise/Tanh";
    layer->_alpha = 0.0;
    layer->_beta = 0.0;
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.46211715, 0.001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    layer->_function = "Elementwise/Logistic";
    layer->_alpha = 0.0;
    layer->_beta = 0.0;
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.62245933120, 0.001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    layer->_function = "Elementwise/SoftSign";
    layer->_alpha = 0.0;
    layer->_beta = 0.0;
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 0.333333333, 0.001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    layer->_function = "Softmax";
    layer->_alpha = 0.0;
    layer->_beta = 0.0;
    ASSERT_NO_THROW(layer->createForwardPipeline());
    ASSERT_NO_THROW(nn->forward({{{0.5}}}));
    ASSERT_NEAR(nn->_pipelines[0][0]._outputValues[0][0], 1.0, 0.001);
    ASSERT_NO_THROW(nn->backward({{{0.5}}}));

    knlohmann::json baseLayerJs = layerJs;

    layerJs = baseLayerJs;
    layerJs.erase("Function");
    ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

    layerJs = baseLayerJs;
    layerJs["Function"] = 1.0;
    ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

    layerJs = baseLayerJs;
    layerJs["Function"] = "Unknown";
    ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

    layerJs = baseLayerJs;
    layerJs["Function"] = "Softmax";
    ASSERT_NO_THROW(layer->setConfiguration(layerJs));

    layerJs = baseLayerJs;
    layerJs.erase("Alpha");
    ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

    layerJs = baseLayerJs;
    layerJs["Alpha"] = "Not a Number";
    ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

    layerJs = baseLayerJs;
    layerJs["Alpha"] = 1.0;
    ASSERT_NO_THROW(layer->setConfiguration(layerJs));

    layerJs = baseLayerJs;
    layerJs.erase("Beta");
    ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

    layerJs = baseLayerJs;
    layerJs["Beta"] = "Not a Number";
    ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

    layerJs = baseLayerJs;
    layerJs["Beta"] = 1.0;
    ASSERT_NO_THROW(layer->setConfiguration(layerJs));

    delete layer;
   }

  TEST(NeuralNetwork, InputLayerKorali)
  {
   Experiment e;
   e._logger = new Logger("Detailed", stdout);

   NeuralNetwork* nn;
   knlohmann::json neuralNetworkConfig;
   neuralNetworkConfig["Type"] = "Neural Network";
   neuralNetworkConfig["Engine"] = "Korali";
   neuralNetworkConfig["Timestep Count"] = 1;
   neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
   neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
   neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Activation";
   neuralNetworkConfig["Layers"][1]["Function"] = "Elementwise/Linear";
   neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
   neuralNetworkConfig["Mode"] = "Training";

   ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
   ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
   auto baseNNJs = neuralNetworkConfig;

   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
   ASSERT_NO_THROW(nn->applyVariableDefaults());
   ASSERT_NO_THROW(nn->initialize());

   Input* layer = dynamic_cast<Input*>(nn->_pipelines[0][0]._layerVector[0]);
   ASSERT_NO_THROW(layer->applyVariableDefaults());
   knlohmann::json layerJs;
   ASSERT_NO_THROW(layer->getConfiguration(layerJs));

   nn->_mode = "Inference";
   ASSERT_ANY_THROW(layer->backwardData(0));
   ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
   ASSERT_ANY_THROW(layer->createBackwardPipeline());
   nn->_mode = "Training";
   ASSERT_NO_THROW(layer->backwardData(0));
   ASSERT_NO_THROW(layer->backwardHyperparameters(0));
   ASSERT_NO_THROW(layer->createBackwardPipeline());
  }

  TEST(NeuralNetwork, OutputLayerKorali)
  {
   Experiment e;
   e._logger = new Logger("Detailed", stdout);

   NeuralNetwork* nn;
   knlohmann::json neuralNetworkConfig;
   neuralNetworkConfig["Type"] = "Neural Network";
   neuralNetworkConfig["Engine"] = "Korali";
   neuralNetworkConfig["Timestep Count"] = 1;
   neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
   neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
   neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Output";
   neuralNetworkConfig["Mode"] = "Training";

   ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
   ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
   auto baseNNJs = neuralNetworkConfig;

   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
   ASSERT_NO_THROW(nn->applyVariableDefaults());
   ASSERT_NO_THROW(nn->initialize());

   Output* layer = dynamic_cast<Output*>(nn->_pipelines[0][0]._layerVector[1]);
   ASSERT_NO_THROW(layer->applyVariableDefaults());
   knlohmann::json layerJs;
   ASSERT_NO_THROW(layer->getConfiguration(layerJs));

   nn->_mode = "Inference";
   ASSERT_ANY_THROW(layer->backwardData(0));
   ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
   ASSERT_ANY_THROW(layer->createBackwardPipeline());
   nn->_mode = "Training";
   ASSERT_NO_THROW(layer->backwardData(0));
   ASSERT_NO_THROW(layer->backwardHyperparameters(0));
   ASSERT_NO_THROW(layer->createBackwardPipeline());

   ASSERT_NO_THROW(layer->getOutput());

   layer->_index = 0;
   ASSERT_ANY_THROW(layer->initialize());
   layer->_index = nn->_layers.size()-1;
   ASSERT_NO_THROW(layer->initialize());

   layer->_scale = std::vector<float>({1.0, 1.0});
   ASSERT_ANY_THROW(layer->initialize());
   layer->_scale = std::vector<float>({1.0});
   ASSERT_NO_THROW(layer->initialize());

   layer->_shift = std::vector<float>({1.0, 1.0});
   ASSERT_ANY_THROW(layer->initialize());
   layer->_shift = std::vector<float>({1.0});
   ASSERT_NO_THROW(layer->initialize());

   layer->_transformationMask = std::vector<std::string>({"Identity", "Identity"});
   ASSERT_ANY_THROW(layer->initialize());
   layer->_transformationMask = std::vector<std::string>({"Unknown"});
   ASSERT_ANY_THROW(layer->initialize());
   layer->_transformationMask = std::vector<std::string>({"Identity"});
   ASSERT_NO_THROW(layer->initialize());
   layer->_transformationMask = std::vector<std::string>({"Absolute"});
   ASSERT_NO_THROW(layer->initialize());
   layer->_transformationMask = std::vector<std::string>({"Softplus"});
   ASSERT_NO_THROW(layer->initialize());
   layer->_transformationMask = std::vector<std::string>({"Tanh"});
   ASSERT_NO_THROW(layer->initialize());
   ASSERT_NO_THROW(layer->initialize());
   layer->_transformationMask = std::vector<std::string>({"Sigmoid"});
   ASSERT_NO_THROW(layer->initialize());

   knlohmann::json baseLayerJs = layerJs;

   layerJs = baseLayerJs;
   layerJs.erase("Output Channels");
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Output Channels"] = "Not a Number";
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Output Channels"] = 1;
   ASSERT_NO_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs.erase("Weight Scaling");
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Weight Scaling"] = "Not a Number";
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Weight Scaling"] = 1.0;
   ASSERT_NO_THROW(layer->setConfiguration(layerJs));
  }

  TEST(NeuralNetwork, LinearLayerKorali)
  {
   Experiment e;
   e._logger = new Logger("Detailed", stdout);

   NeuralNetwork* nn;
   knlohmann::json neuralNetworkConfig;
   neuralNetworkConfig["Type"] = "Neural Network";
   neuralNetworkConfig["Engine"] = "Korali";
   neuralNetworkConfig["Timestep Count"] = 1;
   neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
   neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
   neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Linear";
   neuralNetworkConfig["Layers"][1]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
   neuralNetworkConfig["Mode"] = "Training";

   ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
   ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
   auto baseNNJs = neuralNetworkConfig;

   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
   ASSERT_NO_THROW(nn->applyVariableDefaults());
   ASSERT_NO_THROW(nn->initialize());

   Linear* layer = dynamic_cast<Linear*>(nn->_pipelines[0][0]._layerVector[1]);
   ASSERT_NO_THROW(layer->applyVariableDefaults());
   knlohmann::json layerJs;
   ASSERT_NO_THROW(layer->getConfiguration(layerJs));

   nn->_mode = "Inference";
   ASSERT_ANY_THROW(layer->backwardData(0));
   ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
   ASSERT_ANY_THROW(layer->createBackwardPipeline());
   nn->_mode = "Training";
   ASSERT_NO_THROW(layer->backwardData(0));
   ASSERT_NO_THROW(layer->backwardHyperparameters(0));
   ASSERT_NO_THROW(layer->createBackwardPipeline());
  }

  TEST(NeuralNetwork, GRULayerKorali)
  {
   Experiment e;
   e._logger = new Logger("Detailed", stdout);

   NeuralNetwork* nn;
   knlohmann::json neuralNetworkConfig;
   neuralNetworkConfig["Type"] = "Neural Network";
   neuralNetworkConfig["Engine"] = "Korali";
   neuralNetworkConfig["Timestep Count"] = 1;
   neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
   neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
   neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Recurrent/GRU";
   neuralNetworkConfig["Layers"][1]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
   neuralNetworkConfig["Mode"] = "Training";

   ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
   ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
   auto baseNNJs = neuralNetworkConfig;

   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
   ASSERT_NO_THROW(nn->applyVariableDefaults());

   // Not supported yet
   ASSERT_ANY_THROW(nn->initialize());

   GRU* layer = dynamic_cast<GRU*>(nn->_pipelines[0][0]._layerVector[1]);
   ASSERT_NO_THROW(layer->applyVariableDefaults());
   knlohmann::json layerJs;
   ASSERT_NO_THROW(layer->getConfiguration(layerJs));

   nn->_mode = "Inference";
   ASSERT_ANY_THROW(layer->backwardData(0));
   ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
   ASSERT_ANY_THROW(layer->createBackwardPipeline());
   nn->_mode = "Training";
   ASSERT_NO_THROW(layer->backwardData(0));
   ASSERT_NO_THROW(layer->backwardHyperparameters(0));
   ASSERT_NO_THROW(layer->createBackwardPipeline());
  }

  TEST(NeuralNetwork, LSTMLayerKorali)
  {
   Experiment e;
   e._logger = new Logger("Detailed", stdout);

   NeuralNetwork* nn;
   knlohmann::json neuralNetworkConfig;
   neuralNetworkConfig["Type"] = "Neural Network";
   neuralNetworkConfig["Engine"] = "Korali";
   neuralNetworkConfig["Timestep Count"] = 1;
   neuralNetworkConfig["Batch Sizes"] = std::vector<size_t>({1});
   neuralNetworkConfig["Layers"][0]["Type"] = "Layer/Input";
   neuralNetworkConfig["Layers"][0]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][1]["Type"] = "Layer/Recurrent/LSTM";
   neuralNetworkConfig["Layers"][1]["Output Channels"] = 1;
   neuralNetworkConfig["Layers"][2]["Type"] = "Layer/Output";
   neuralNetworkConfig["Mode"] = "Training";

   ASSERT_NO_THROW(nn = dynamic_cast<NeuralNetwork *>(Module::getModule(neuralNetworkConfig, &e)));
   ASSERT_NO_THROW(nn->applyModuleDefaults(neuralNetworkConfig));
   auto baseNNJs = neuralNetworkConfig;

   ASSERT_NO_THROW(nn->setConfiguration(neuralNetworkConfig));
   ASSERT_NO_THROW(nn->applyVariableDefaults());

   // Not supported yet
   ASSERT_ANY_THROW(nn->initialize());

   LSTM* layer = dynamic_cast<LSTM*>(nn->_pipelines[0][0]._layerVector[1]);
   ASSERT_NO_THROW(layer->applyVariableDefaults());
   knlohmann::json layerJs;
   ASSERT_NO_THROW(layer->getConfiguration(layerJs));

//   ASSERT_NO_THROW(layer->initialize());
//   layer->_depth = 10;
//   layer->_outputChannels = 10;
//   ASSERT_ANY_THROW(layer->initialize());
//   layer->_outputChannels = 1;
//   ASSERT_NO_THROW(layer->initialize());
//   layer->_depth = 1;
//   ASSERT_NO_THROW(layer->initialize());

   nn->_mode = "Inference";
   ASSERT_ANY_THROW(layer->backwardData(0));
   ASSERT_ANY_THROW(layer->backwardHyperparameters(0));
   ASSERT_ANY_THROW(layer->createBackwardPipeline());
   nn->_mode = "Training";
   ASSERT_NO_THROW(layer->backwardData(0));
   ASSERT_NO_THROW(layer->backwardHyperparameters(0));
   ASSERT_NO_THROW(layer->createBackwardPipeline());

   knlohmann::json baseLayerJs = layerJs;

   layerJs = baseLayerJs;
   layerJs.erase("Depth");
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Depth"] = "Not a Number";
   ASSERT_ANY_THROW(layer->setConfiguration(layerJs));

   layerJs = baseLayerJs;
   layerJs["Depth"] = 1;
   ASSERT_NO_THROW(layer->setConfiguration(layerJs));
  }
} // namespace
