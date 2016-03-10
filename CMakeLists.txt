cmake_minimum_required(VERSION 3.3)
project(MyNN)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Net.cpp Net.h Neuron.cpp Neuron.h NeuronConnection.cpp NeuronConnection.h NNData.cpp NNData.h Layer.cpp Layer.h)
add_executable(MyNN ${SOURCE_FILES})