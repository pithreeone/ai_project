#ifndef _DLMATH_H_
#define _DLMATH_H_

#include "Eigen/Dense"
#include <vector>
#include "kernel.h"

namespace DLMATH{
    double Sigmoid(double x);

    double SigmoidPrime(double x);

    // [y1, y2,...,yi] = softmax(a1, a2,...,aj)
    // yi = exp(ai)/(exp(a1)+exp(a2)+...+exp(aj))
    Eigen::VectorXf SoftMax(Eigen::VectorXf x);

    // return dyi/daj
    double SoftMaxPrime(Eigen::VectorXf x, int i, int j);

    double ReLU(double x);

    double ReLUPrime(double x);

    Eigen::VectorXf flatten(std::vector<Eigen::MatrixXf> x);

    std::vector<Eigen::MatrixXf> Conv3d_2d(std::vector<Eigen::MatrixXf> x, std::vector<Eigen::MatrixXf> kernel);
    
    std::vector<Eigen::MatrixXf> Conv3d_3d(std::vector<Eigen::MatrixXf> x, std::vector<Kernel3d> kernels);
}




#endif