#include <iostream>
#include <nn.h>


void NN::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding){
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    kernel_size_ = kernel_size;
    stride_ = stride;
    padding_ = padding;
    for(int i=0; i<out_channels_; i++){
        weights_kernel_.push_back(*(new Kernel3d(kernel_size_, in_channels)));
    }
    // std::cout << weights_kernel_.size() << std::endl;
}

void NN::Linear(int in_channels, int out_channels){
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    weights_fc_.resize(in_channels_ + 1, out_channels_);
}

void NN::MaxPool2d(int max_pool){
    max_pool_ = max_pool;
}

void NN::ReLU(){
    activation_function_ = "ReLU";
};

void NN::Sigmoid(){
    activation_function_ = "Sigmoid";
};

void NN::CrossEntropyLoss(){
    loss_function_ = "CrossEntropyLoss";
};