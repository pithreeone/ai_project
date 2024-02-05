#ifndef _DLMATH_H_
#define _DLMATH_H_

#include "Eigen/Dense"
#include <vector>

namespace DLMATH{
    double Sigmoid(double x);

    double SigmoidPrime(double x);

    Eigen::VectorXf SoftMax(Eigen::VectorXf x);

    double SoftMaxPrime();

    double ReLU(double x);

    double ReLUPrime(double x);

    Eigen::VectorXf flatten(std::vector<Eigen::MatrixXf> x);

    std::vector<Eigen::MatrixXf> Conv3d(std::vector<Eigen::MatrixXf> x, std::vector<Eigen::MatrixXf> kernel);
}




#endif