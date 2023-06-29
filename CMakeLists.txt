#CC Build instructions
#module load python/3.7 and  module load scipy-stack and module load mariadb
#module load mariadb-connector-c

cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

#add_executable(Graph graph_expeirments.cpp include/nn/networks/graph.h include/utils.h src/utils.cpp src/nn/networks/graph.cpp include/experiment/Metric.h include/experiment/Database.h include/experiment/Experiment.h
#        src/experiment/Database.cpp src/experiment/Experiment.cpp src/experiment/Metric.cpp src/nn/networks/graph.cpp src/nn/networks/edge.cpp src/nn/networks/vertex.cpp
#        include/nn/networks/vertex.h include/nn/graphfactory.h include/nn/graphfactory.h src/nn/graphfactory.cpp include/nn/weight_initializer.h src/nn/weight_initializer.cpp include/nn/weight_optimizer.h src/nn/weight_optimizer.cpp
#        include/nn/optimizer_factory.h src/nn/optimizer_factory.cpp include/environments/input_distribution.h
#        src/environments/input_distribution.cpp include/environments/environment_factory.h src/environments/environment_factory.cpp src/environments/mnist_environment.cpp
#        src/nn/networks/graph_utility_prop_implementations.cpp src/nn/architecture_initializer.cpp)

add_executable(FeatureSearch feature_search.cpp include/nn/networks/graph.h include/utils.h src/utils.cpp src/nn/networks/graph.cpp include/experiment/Metric.h include/experiment/Database.h include/experiment/Experiment.h
        src/experiment/Database.cpp src/experiment/Experiment.cpp src/experiment/Metric.cpp src/nn/networks/graph.cpp src/nn/networks/edge.cpp src/nn/networks/vertex.cpp
        include/nn/networks/vertex.h include/nn/graphfactory.h include/nn/graphfactory.h src/nn/graphfactory.cpp include/nn/weight_initializer.h src/nn/weight_initializer.cpp include/nn/weight_optimizer.h src/nn/weight_optimizer.cpp
        include/nn/optimizer_factory.h src/nn/optimizer_factory.cpp include/environments/input_distribution.h
        src/environments/input_distribution.cpp include/environments/environment_factory.h src/environments/environment_factory.cpp src/environments/mnist_environment.cpp
        src/nn/networks/graph_utility_prop_implementations.cpp src/nn/architecture_initializer.cpp)


add_executable(CFS continual_feature_search.cpp include/nn/networks/graph.h include/utils.h src/utils.cpp src/nn/networks/graph.cpp include/experiment/Metric.h include/experiment/Database.h include/experiment/Experiment.h
        src/experiment/Database.cpp src/experiment/Experiment.cpp src/experiment/Metric.cpp src/nn/networks/graph.cpp src/nn/networks/edge.cpp src/nn/networks/vertex.cpp
        include/nn/networks/vertex.h include/nn/graphfactory.h include/nn/graphfactory.h src/nn/graphfactory.cpp include/nn/weight_initializer.h src/nn/weight_initializer.cpp include/nn/weight_optimizer.h src/nn/weight_optimizer.cpp
        include/nn/optimizer_factory.h src/nn/optimizer_factory.cpp include/environments/input_distribution.h
        src/environments/input_distribution.cpp include/environments/environment_factory.h src/environments/environment_factory.cpp src/environments/mnist_environment.cpp
        src/nn/networks/graph_utility_prop_implementations.cpp src/nn/architecture_initializer.cpp)


add_executable(LoadModel load_model.cpp include/nn/networks/graph.h include/utils.h src/utils.cpp src/nn/networks/graph.cpp include/experiment/Metric.h include/experiment/Database.h include/experiment/Experiment.h
        src/experiment/Database.cpp src/experiment/Experiment.cpp src/experiment/Metric.cpp src/nn/networks/graph.cpp src/nn/networks/edge.cpp src/nn/networks/vertex.cpp
        include/nn/networks/vertex.h include/nn/graphfactory.h include/nn/graphfactory.h src/nn/graphfactory.cpp include/nn/weight_initializer.h src/nn/weight_initializer.cpp include/nn/weight_optimizer.h src/nn/weight_optimizer.cpp
        include/nn/optimizer_factory.h src/nn/optimizer_factory.cpp include/environments/input_distribution.h
        src/environments/input_distribution.cpp include/environments/environment_factory.h src/environments/environment_factory.cpp src/environments/mnist_environment.cpp
        src/nn/networks/graph_utility_prop_implementations.cpp src/nn/architecture_initializer.cpp)


add_executable(Count include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        count_configs.cpp include/nn/architure_initializer.h)

#target_link_libraries(Graph PUBLIC tbb)
#target_link_libraries(Graph PUBLIC mariadb)

target_link_libraries(FeatureSearch PUBLIC tbb)
target_link_libraries(FeatureSearch PUBLIC mariadb)

target_link_libraries(CFS PUBLIC tbb)
target_link_libraries(CFS PUBLIC mariadb)

target_link_libraries(LoadModel PUBLIC tbb)
target_link_libraries(LoadModel PUBLIC mariadb)
#

target_link_libraries(Count PUBLIC tbb)
target_link_libraries(Count PUBLIC mariadb)
##

project(FlexibleNN LANGUAGES CXX)






