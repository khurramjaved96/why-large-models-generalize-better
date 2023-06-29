//
// Created by Khurram Javed on 2023-03-12.
//

#include "include/environments/input_distribution.h"
#include "include/nn/networks/graph.h"
#include "include/nn/networks/vertex.h"
#include <iostream>
#include <random>
#include <vector>

#include "include/environments/environment_factory.h"
#include "include/environments/mnist/mnist_reader.hpp"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include "include/nn/architure_initializer.h"
#include "include/nn/graphfactory.h"
#include "include/nn/optimizer_factory.h"
#include "include/nn/weight_initializer.h"
#include "include/nn/weight_optimizer.h"
#include "include/utils.h"
#include <random>
#include <string>

int main(int argc, char *argv[]) {
  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  Metric error_metric = Metric(
      my_experiment->database_name, "error",
      std::vector<std::string>{"run", "step", "train_error", "train_accuracy",
                               "test_error", "test_accuracy"},
      std::vector<std::string>{"int", "int", "real", "real", "real", "real"},
      std::vector<std::string>{"run", "step"});

  MNISTEnviroment *env = new MNISTEnviroment(0);
  MNISTTestEnviroment *test_env = new MNISTTestEnviroment(0);
  int win = 0;
  float min_error = 1000;
  float min_seed = -10;

  Graph *network = GraphFactory::get_graph("", my_experiment, 0);

  auto network_initializer = ArchitectureInitializer();
  auto seeds_f = my_experiment->get_vector_of_floats("features");
  std::vector<int> features;
  for (auto f : seeds_f) {
    features.push_back(int(f));
  }
  std::cout << "Seeds\n";
  print_vector(features);
  if(my_experiment->get_string_param("network") == "linear_learning") {
    network =
        network_initializer.initialize_linear_learning_network_list_of_seeds(
            network, my_experiment->get_string_param("non_linearity"), my_experiment->get_float_param("step_size") ,my_experiment->get_float_param("lower"), my_experiment->get_float_param("higher"),
            features);
  }
  else if(my_experiment->get_string_param("network") == "fixed_features"){
    network =
        network_initializer.initialize_fixed_features_network_list_of_seeds(
            network, my_experiment->get_string_param("non_linearity"), my_experiment->get_float_param("step_size") ,my_experiment->get_float_param("lower"), my_experiment->get_float_param("higher"),
            features);
  }

  Optimizer *opti = OptimizerFactory::get_optimizer(network, my_experiment);

  float train_error = 5;
  bool check_flag = false;
  float THRESHOLD = my_experiment->get_float_param("threshold");
  for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {
    env->step();
    auto inps = env->get_features();
    float target = env->get_target();
    network->set_input_values(inps);
    float prediction = network->update_values();
    network->estimate_gradient(target);
    opti->update_weights(network);

    if (i % my_experiment->get_int_param("frequency") ==
            my_experiment->get_int_param("frequency") - 1 ||
        check_flag) {
      check_flag = false;
      //      Evaluate error on the train set
      float correct = 0;
      auto list_of_x = env->get_all_x();
      auto list_of_targets = env->get_all_y();
      float total_error = 0;
      for (int temp = 0; temp < list_of_x.size(); temp++) {
        auto temp_x = list_of_x[temp];
        float temp_target = list_of_targets[temp];
        network->set_input_values(temp_x);
        float temp_prediction = network->update_values();
        int target = int(list_of_targets[temp]);
        total_error += -log(network->prediction_probabilites[target]);
        if (temp_prediction == target)
          correct++;
      }
      total_error /= list_of_x.size();
      train_error = total_error;

      //        std::cout << "Actual error = " << total_error << std::endl;
      //      std::cout << "Accuracy " << correct / list_of_x.size() <<
      //      std::endl;
      std::vector<std::string> val;
      val.push_back(std::to_string(my_experiment->get_int_param("run")));
      val.push_back(std::to_string(i));
      val.push_back(std::to_string(total_error));
      val.push_back(std::to_string(correct / list_of_x.size()));

      correct = 0;
      list_of_x = test_env->get_all_x();
      list_of_targets = test_env->get_all_y();
      total_error = 0;
      for (int temp = 0; temp < list_of_x.size(); temp++) {
        auto temp_x = list_of_x[temp];
        float temp_target = list_of_targets[temp];
        network->set_input_values(temp_x);
        float temp_prediction = network->update_values();
        int target = int(list_of_targets[temp]);
        total_error += -log(network->prediction_probabilites[target]);
        if (temp_prediction == target)
          correct++;
      }
      total_error /= list_of_x.size();
      //        std::cout << "Test error = " << total_error << std::endl;
      val.push_back(std::to_string(total_error));
      val.push_back(std::to_string(correct / list_of_x.size()));
      error_metric.record_value(val);
      error_metric.commit_values();
    }
  }
  std::cout << train_error << std::endl;
  error_metric.commit_values();
  //
}
