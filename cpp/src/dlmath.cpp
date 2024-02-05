#include "dlmath.h" // Replace with the actual header file name
#include <cmath>
#include <iostream>
namespace DLMATH {

    double Sigmoid(double x){
        return 1/(1+exp(-x));
    }

    double SigmoidPrime(double x){
        return exp(x)/pow(1+exp(x),2);
    }

    Eigen::VectorXf SoftMax(Eigen::VectorXf x){
        double sum = 0;
        for (auto i=0; i<x.size(); i++){
            sum += exp(x[i]);
        }
        Eigen::VectorXf y;
        for (auto i=0; i<x.size(); i++){
            y[i] = exp(x[i])/sum;
        }
        return y;
    }

    double SoftMaxPrime(Eigen::VectorXf x, int i, int j){
        Eigen::VectorXf y = SoftMax(x);
        if (i == j){
            return y[i]*(1-y[j]);
        }else{
            return -y[i]*y[j];
        }
    }

    double ReLU(double x){
        if (x > 0){
            return x;
        }else{
            return 0;
        }
    }

    double ReLUPrime(double x){
        if (x > 0){
            return 1;
        }else{
            return 0;
        }
    }

    Eigen::VectorXf flatten(std::vector<Eigen::MatrixXf> x){
        int l = 0;
        Eigen::VectorXf y;
        for (auto i=0; i<x.size(); i++){
            for (auto j=0; j<x[i].rows(); j++){
                for (auto k=0; k<x[i].cols(); k++){
                    y(l) = x[i](j,k);
                    l++;
                }
            }
        }
        return y;
    }

    std::vector<Eigen::MatrixXf> Conv3d_2d(std::vector<Eigen::MatrixXf> x, std::vector<Eigen::MatrixXf> kernel){
    
    }

    std::vector<Eigen::MatrixXf> Conv3d_3d(std::vector<Eigen::MatrixXf> x, std::vector<Kernel3d> kernels){
        
    }
}