#ifndef _NN_H_
#define _NN_H_

#include <string>
#include <vector>
#include <Eigen/Dense>

using namespace std;

class NN{
private:
    string function_type_;  // [Conv2d, Linear, MaxPool2d, LossFunction]
    uint in_channels_;
    uint out_channels_;
    uint kernel_size_;
    uint stride_;
    uint padding_;
    bool bias_;
    string activation_function_;
    string loss_function_;

public:

    // Set an convolutional layer
    void Conv2d(uint in_channels, uint out_channels, uint kernel_size, uint stride, uint padding);
    
    // Set an linear layer
    void Linear(uint in_channels, uint out_channels);

    // Set MaxPool layer
    void MaxPool2d(uint max_pool);

    // Set an activation function
    void ReLU();
    void Sigmoid();

    // Set an loss function
    void CrossEntropyLoss();




};

#endif