cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

# Use C++17 or later if needed
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add your source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")

# Define your executable
add_executable(neuralNetwork ${SOURCES})