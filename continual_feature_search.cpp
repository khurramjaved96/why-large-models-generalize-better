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

  int USED = 1;
  int REJECTED = 2;
  int UNEXPLORED = 3;
  int COMMITTED = 4;

  std::vector<int> all_features;
  for(int i = 0; i < 100000; i++){
    all_features.push_back(UNEXPLORED);
  }

  Graph *network = GraphFactory::get_graph("", my_experiment, 0);

  auto network_initializer = ArchitectureInitializer();
//  auto seeds_f = my_experiment->get_int_param("features");
  std::vector<int> features;
  std::mt19937 mt(my_experiment->get_int_param("seed"));
  std::uniform_int_distribution<int> seed_sampler(0, 100000-1);
  for (int f = 0; f< my_experiment->get_int_param("features"); f++) {
    features.push_back(f);
    all_features[f] = USED;
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
  int size = 0;
  float train_error = 5;
  bool check_flag = false;
  int freq = my_experiment->get_int_param("frequency");
  float THRESHOLD = my_experiment->get_float_param("threshold");
  for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {
//    std::cout << i << "\n";
    env->step();
    auto inps = env->get_features();
    float target = env->get_target();
    network->set_input_values(inps);
    float prediction = network->update_values();
    network->update_utility();
    network->estimate_gradient(target);
    opti->update_weights(network);
    if(i%10000==9999){
      std::cout << "Step = " << i << "\n";
    }

    if(i%(freq)==(freq - 1)){
      std::vector<float> traces;
      std::vector<float> traces_2;
      float max = -1;
      int max_index = -1;
      int max_seed = -1;
      int uncommitted = 0;
      for(int k = 784; k < 784 + my_experiment->get_int_param("features"); k++){
        ReluVertex* r = static_cast<ReluVertex*>(network->list_of_vertices[k]);
        traces.push_back(network->list_of_vertices[k]->utility_trace);
        traces_2.push_back(network->list_of_vertices[k]->utility_trace);
      }
      std::sort(traces_2.begin(), traces_2.end());
      float fifty = traces_2[traces_2.size()/5];
      std::cout << "Fifty = " << fifty << std::endl;
      for(int i = 0; i < features.size(); i++){
//        std::cout << traces[i] << " " << fifty << std::endl;
        if(traces[i] <= fifty){
          std::cout << "Changing feature " << features[i] << std::endl;
          features[i] = seed_sampler(mt);
        }
      }
      network = GraphFactory::get_graph("", my_experiment, 0);
              network =
                  network_initializer.initialize_fixed_features_network_list_of_seeds(
                      network, my_experiment->get_string_param("non_linearity"),
                      my_experiment->get_float_param("step_size"),
                      my_experiment->get_float_param("lower"),
                      my_experiment->get_float_param("higher"), features);
              opti = OptimizerFactory::get_optimizer(network, my_experiment);

              env = new MNISTEnviroment(0);
              test_env = new MNISTTestEnviroment(0);
        i = 0;
        freq += 30000;
        if(freq > 300000)
          freq = 300000;

//      print_vector(traces);

//      std::cout << "Max val = " << max << std::endl;
//      std::cout << "Max seed = " << max_seed << std::endl;
//      all_features[max_seed]= COMMITTED;
//      for(int i = 0; i < features.size(); i++){
//        if(all_features[features[i]] != COMMITTED){
//          features[i] = seed_sampler(mt);
//        }
////        print_vector(features);
//      }
//      if(uncommitted > 0) {
//        network = GraphFactory::get_graph("", my_experiment, 0);
//        network =
//            network_initializer.initialize_fixed_features_network_list_of_seeds(
//                network, my_experiment->get_string_param("non_linearity"),
//                my_experiment->get_float_param("step_size"),
//                my_experiment->get_float_param("lower"),
//                my_experiment->get_float_param("higher"), features);
//        opti = OptimizerFactory::get_optimizer(network, my_experiment);
//        std::cout << "Total uncommited = " << uncommitted << std::endl;
//      }
//      int total_reset = 0;
//      for(int k = 784; k < 784 + my_experiment->get_int_param("features"); k++) {
//        ReluVertex *r = static_cast<ReluVertex *>(network->list_of_vertices[k]);
//        if(all_features[r->seed] != COMMITTED){
//          int seed;
////          all_features[r->seed] = UNEXPLORED;
//          seed  = seed_sampler(mt);
////          std::cout << "New seed = " << seed << std::endl;
////          all_features[seed] = USED;
//          network = update_feature(network, k, seed,  my_experiment->get_float_param("step_size") ,my_experiment->get_float_param("lower"), my_experiment->get_float_param("higher"));
//          total_reset++;
//        }
//        else{
//          std::cout << "Not resetting seed " << r->seed << " index " << k << std::endl;
//        }
//
//      }
//      std::cout << "Total reset = " << total_reset << std::endl;
    }
//    Resetting weights

    if (i % my_experiment->get_int_param("frequency") ==
        my_experiment->get_int_param("frequency")- 2) {
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

              std::cout << "Actual error = " << total_error << std::endl;
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
              std::cout << "Test error = " << total_error << std::endl;
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
