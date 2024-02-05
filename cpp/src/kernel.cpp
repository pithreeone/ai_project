#include "kernel.h"
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

Kernel3d::Kernel3d(){
}


Kernel3d::Kernel3d(int kernel_size, int channels){
    kernel_size_ = kernel_size;
    channels_ = channels;

    kernel_.resize(channels_);
    kernel_derivative_.resize(channels_);

    for (auto i=kernel_.begin(); i!=kernel_.end(); i++){
        i->resize(kernel_size_, kernel_size_);
    }
    for (auto i=kernel_derivative_.begin(); i!=kernel_derivative_.end(); i++){
        i->resize(kernel_size_, kernel_size_);
    }
}

void Kernel3d::randomIntialize(double min, double max){
    double range = max-min;
    for (auto i=kernel_.begin(); i!=kernel_.end(); i++){
        MatrixXf m = MatrixXf::Random(kernel_size_, kernel_size_);
        m = (m + MatrixXf::Constant(kernel_size_,kernel_size_,1.))*range/2.;
        *i = (m + MatrixXf::Constant(kernel_size_,kernel_size_,min));
    }
    for (auto i=kernel_derivative_.begin(); i!=kernel_derivative_.end(); i++){
        MatrixXf m = MatrixXf::Random(kernel_size_, kernel_size_);
        m = (m + MatrixXf::Constant(kernel_size_,kernel_size_,1.))*range/2.;
        *i = (m + MatrixXf::Constant(kernel_size_,kernel_size_,min));
    }
    for (auto i=0; i!=channels_; i++){
        double f = (double)rand() / RAND_MAX;
        bias_ = min + f * range;
    }
    for (auto i=0; i!=channels_; i++){
        double f = (double)rand() / RAND_MAX;
        bias_derivative_ = min + f * range;
    }
}