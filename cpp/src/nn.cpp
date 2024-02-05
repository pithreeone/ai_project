#include <iostream>
#include "nn.h"


void NN::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding){
    function_type_ = "Conv2d";
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

vector<Eigen::MatrixXf> NN::Conv2d(vector<Eigen::MatrixXf> input){

}

void NN::Linear(int in_channels, int out_channels){
    function_type_ = "Linear";
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    weights_fc_.resize(in_channels_ + 1, out_channels_);
}

vector<Eigen::MatrixXf> NN::Linear(vector<Eigen::MatrixXf> input){

}

void NN::MaxPool2d(int max_pool){
    function_type_ = "MaxPool2d";
    max_pool_ = max_pool;
}

vector<Eigen::MatrixXf> NN::MaxPool2d(vector<Eigen::MatrixXf> input){

}

void NN::ReLU(){
    function_type_ = "Activation";
    activation_function_ = "ReLU";
};

vector<Eigen::MatrixXf> NN::ReLU(vector<Eigen::MatrixXf> input){

}

void NN::Sigmoid(){
    function_type_ = "Activation";
    activation_function_ = "Sigmoid";
};

vector<Eigen::MatrixXf> NN::Sigmoid(vector<Eigen::MatrixXf> input){

}

void NN::Softmax(){
    function_type_ = "Activation";
    activation_function_ = "Softmax";
}

Eigen::VectorXf NN::Softmax(Eigen::VectorXf input){

}

void NN::CrossEntropyLoss(){
    function_type_ = "LossFunction";
    loss_function_ = "CrossEntropyLoss";
};