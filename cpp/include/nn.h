#ifndef _NN_H_
#define _NN_H_

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "kernel.h"

using namespace std;

class NN{
private:
    string function_type_;  // [Conv2d, Linear, MaxPool2d, Activation, LossFunction]
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool bias_;
    int max_pool_;
    string activation_function_; // [ReLU, Sigmoid, Softmax]
    string loss_function_;  // [CrossEntropyLoss]
    
    // weights
    vector<Kernel3d> weights_kernel_;     // In cnn layer
    Eigen::MatrixXf weights_fc_;  // In fully connected layer

    // output of the layer
    vector<Eigen::MatrixXf> output_;

public:

    string getFunctionType(){ return function_type_;}

    string getActivationFunction(){ return activation_function_;}

    string getLossFunction(){ return loss_function_;}

    // The following function has two types. 
    // 1. One is used to define the layer. 
    // 2. Another is used to calculate the output pass through the layer.

    // Set an convolutional layer
    void Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding);
    // Implement the Conv2d math function
    vector<Eigen::MatrixXf> Conv2d(vector<Eigen::MatrixXf> input);
    
    // Set an linear layer
    void Linear(int in_channels, int out_channels);
    // Implement the Fully-connected Linear layer
    vector<Eigen::MatrixXf> Linear(vector<Eigen::MatrixXf> input);

    // Set MaxPool layer
    void MaxPool2d(int max_pool);
    // Implement the MaxPool2d layer
    vector<Eigen::MatrixXf> MaxPool2d(vector<Eigen::MatrixXf> input);


    // Set an activation function
    void ReLU();
    vector<Eigen::MatrixXf> ReLU(vector<Eigen::MatrixXf> input);
    
    void Sigmoid();
    vector<Eigen::MatrixXf> Sigmoid(vector<Eigen::MatrixXf> input);
    
    void Softmax();
    Eigen::VectorXf Softmax(Eigen::VectorXf input);

    // Set an loss function
    void CrossEntropyLoss();




};

#endif