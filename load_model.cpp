//
// Created by Khurram Javed on 2022-08-30.
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
#include <fstream>

int main(int argc, char *argv[]) {
  int a, b;
  float c;
  std::string type;
  Graph *network =
      new UtilityPropagation(28*28,
                             0,
                            0.99999);

  std::ifstream infile("model");
  int old_b = -1;
  while (infile >> a >> b >> c >> type)
  {

    if(b != old_b){
      network->list_of_vertices.push_back(VertexFactory::get_vertex(type));
    }
    old_b = b;
  }
//  network->print_graph();
  std::ifstream infile2("model");
  while (infile2 >> a >> b >> c >> type)
  {
    network->add_edge(c, a, b, 1e-4);
  }

  std::cout << "Model loaded\n";
  MNISTTestEnviroment *env =
      new MNISTTestEnviroment(0);
  float correct = 0;
  auto list_of_x = env->get_all_x();
  auto list_of_targets = env->get_all_y();
  float total_error = 0;
  for (int temp = 0; temp < list_of_x.size(); temp++) {
    auto temp_x = list_of_x[temp];
    float temp_target = list_of_targets[temp];
//    print_vector(temp_x);
    network->set_input_values(temp_x);
    float temp_prediction = network->update_values();
    float temp_delta = temp_prediction - temp_target;
    if (std::abs(temp_delta) < 0.5)
      correct++;

    total_error += temp_delta * temp_delta;
  }
  total_error /= list_of_x.size();
                                  std::cout << "MSE = " << total_error << std::endl;
                                  std::cout << "Accuracy = " << correct / list_of_x.size() << std::endl;
}
