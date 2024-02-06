#ifndef _FCWEIGHT_H_
#define _FCWEIGHT_H_

#include <vector>
#include "Eigen/Dense"

// Handle the weights of the linear layer (fully connected layer)
class FCWeight{
public:
    int in_channels_;
    int out_channels_;

    Eigen::MatrixXf weights_;
    Eigen::MatrixXf weights_derivative_;

    FCWeight();
    FCWeight(int in_channels, int out_channels);

    void resize(int in_channels, int out_channels);

    void zero_grad();
};

#endif