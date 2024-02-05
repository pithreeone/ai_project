cmake_minimum_required(VERSION 3.0)
PROJECT (ai_project)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})


# Add an executable
add_executable (main src/main.cpp src/cnn.cpp src/nn.cpp src/dataLoader.cpp src/optimization.cpp src/kernel.cpp src/dlmath.cpp)

# Specify the include directories
target_include_directories(main PRIVATE include/)
target_link_libraries(main ${OpenCV_LIBS} )


add_executable(DisplayImage src/DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )