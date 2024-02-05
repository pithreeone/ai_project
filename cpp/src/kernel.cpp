#include "kernel.h"


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

}