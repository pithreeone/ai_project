#ifndef _NN_H_
#define _NN_H_

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "kernel.h"

using namespace std;

class NN{
private:
    string function_type_;  // [Conv2d, Linear, MaxPool2d, LossFunction]
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool bias_;
    int max_pool_;
    string activation_function_;
    string loss_function_;  // [CrossEntropyLoss]
    
    // weights
    vector<Kernel3d> weights_kernel_;     // In cnn layer
    Eigen::MatrixXf weights_fc_;  // In fully connected layer

public:

    // Set an convolutional layer
    void Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding);
    
    // Set an linear layer
    void Linear(int in_channels, int out_channels);

    // Set MaxPool layer
    void MaxPool2d(int max_pool);

    // Set an activation function
    void ReLU();
    void Sigmoid();

    // Set an loss function
    void CrossEntropyLoss();




};

#endif