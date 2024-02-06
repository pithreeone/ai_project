#include <iostream>
#include <cfloat>
#include "nn.h"
#include "dlmath.h"


void NN::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding){
    function_type_ = "Conv2d";
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    kernel_size_ = kernel_size;
    stride_ = stride;
    padding_ = padding;

    weights_kernel_ = new Kernel4d(kernel_size_, in_channels_, out_channels_);
    
    // std::cout << weights_kernel_.size() << std::endl;
}

std::vector<Eigen::MatrixXf> NN::Conv2d(std::vector<Eigen::MatrixXf> input){
    // return DLMATH::Conv3d_3d(input, weights_kernel_);
}

void NN::Linear(int in_channels, int out_channels){
    function_type_ = "Linear";
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    weights_linear_ = new FCWeight(in_channels_, out_channels_);
    
}

Eigen::VectorXf NN::Linear(Eigen::VectorXf input){

}

void NN::MaxPool2d(int max_pool){
    function_type_ = "MaxPool2d";
    max_pool_ = max_pool;
}

std::vector<Eigen::MatrixXf> NN::MaxPool2d(std::vector<Eigen::MatrixXf> input){
    int row_input = input[0].rows();
    int col_input = input[0].cols();
    // std::cout << "row: " << row_input << ", col: " << col_input << std::endl;
    std::vector<Eigen::MatrixXf> output;

    for(auto i=0; i<input.size(); i++){
        int row_output = (row_input + max_pool_ - 1)/max_pool_;
        int col_output = (col_input + max_pool_ - 1)/max_pool_;
        Eigen::MatrixXf temp(row_output, col_output);
        for(int j=0; j<row_output; j++){
            for(int k=0; k<col_output; k++){
                // find the maximum value in the square-region
                double max = -DBL_MAX;
                for(int m=0; m<max_pool_*max_pool_; m++){
                    int r = max_pool_ * j + m / max_pool_;
                    int c = max_pool_ * k + m % max_pool_;
                    if(r >= row_input || c >= col_input) { continue; }

                    // std::cout << "r: " << r << ", c: " << c << std::endl;
                    if(input[i](r, c) > max){
                        max = input[i](r, c);
                    }
                }
                temp(j, k) = max;
            }
        }
        output.push_back(temp);
        // std::cout << "row: " << temp.rows() << ", col: " << temp.cols() << std::endl;
    }
    output_mat_.push_back(output);
    
    return output;
}

void NN::ReLU(){
    function_type_ = "Activation";
    activation_function_ = "ReLU";
};

std::vector<Eigen::MatrixXf> NN::ReLU(std::vector<Eigen::MatrixXf> input){
    
}

Eigen::VectorXf NN::ReLU(Eigen::VectorXf input){

}

void NN::Sigmoid(){
    function_type_ = "Activation";
    activation_function_ = "Sigmoid";
};

std::vector<Eigen::MatrixXf> NN::Sigmoid(std::vector<Eigen::MatrixXf> input){

}

Eigen::VectorXf NN::Sigmoid(Eigen::VectorXf input){

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