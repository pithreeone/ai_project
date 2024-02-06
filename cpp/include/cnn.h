#ifndef _CNN_H_
#define _CNN_H_

#include <Eigen/Dense>
#include <string>
#include <vector>
#include "nn.h"


class CNN{
public:
    
    bool network_is_legal_;
    std::vector<NN> network_;
    Eigen::VectorXf output_;

    // Constructor: set the structure of neural network
    CNN();
    
    // Print the network of CNN
    void print();

    // check if network is legal, change the variable: network_is_legal_
    void checkNetwork();

    // do the forward-propagation with just one input data
    void forward(Eigen::MatrixXf input, Eigen::VectorXf& output, int& y);

    // do the forward-propagation with many input data
    void forward(std::vector<Eigen::MatrixXf> input, std::vector<Eigen::VectorXf>& output, std::vector<int>& y);

};

#endif