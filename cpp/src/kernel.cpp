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
        // Matrix filled with random numbers between (-1,1)
        MatrixXf m = MatrixXf::Random(kernel_size_, kernel_size_);
        // adjust the range to (0,1)
        m = (m + MatrixXf::Constant(kernel_size_,kernel_size_,1.))*range/2.;
        // *i means kernel_, adjust the range to (min,max)
        *i = (m + MatrixXf::Constant(kernel_size_,kernel_size_,min));
    }
    for (auto i=0; i!=channels_; i++){
        // f is random double number between (0,1)
        double f = (double)rand() / RAND_MAX;
        // adjust the range to (min,max)
        bias_ = min + f * range;
    }
}

Kernel4d::Kernel4d(int kernel_size, int in_channels, int out_channels){
    kernel_size_ = kernel_size;
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    for(int i=0; i<out_channels_; i++){
        kernels_.push_back(*(new Kernel3d(kernel_size_, in_channels)));
    }
}

void Kernel4d::zero_grad(){
    
}