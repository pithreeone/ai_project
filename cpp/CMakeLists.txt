cmake_minimum_required(VERSION 3.0)
PROJECT (ai_project)

# Add an executable
ADD_EXECUTABLE (${PROJECT_NAME} src/main.cpp src/cnn.cpp)

# Specify the include directories
target_include_directories(${PROJECT_NAME} PRIVATE include/)