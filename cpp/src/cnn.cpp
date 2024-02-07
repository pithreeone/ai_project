#include <iostream>
#include "cnn.h"
#include "dlmath.h"


// The class define the neural network structure, 
// using the user-defined class: NN, which is defined in nn.h


CNN::CNN(){
    std::cout << "Initialize network structure of CNN" << std::endl;

    // Hidden Layer 1
    NN layer1, layer1_activation, layer1_maxpool;
    layer1.Conv2d(1, 16, 5, 1, 2);
    layer1_activation.ReLU();
    layer1_maxpool.MaxPool2d(2);

    // Hidden Layer 2
    NN layer2, layer2_activation, layer2_maxpool;
    layer2.Conv2d(16, 32, 5, 1, 2);
    layer2_activation.ReLU();
    layer2_maxpool.MaxPool2d(2);

    // Flatten Layer
    NN layer_flatten;
    // layer_flatten.Flatten();

    // Fully-connected Layer
    NN layer3, layer3_activation;
    layer3.Linear(7*7*32, 10);
    layer3_activation.Softmax();


    // push the above layer to the variable: network
    network_.push_back(layer1);
    network_.push_back(layer1_activation);
    network_.push_back(layer1_maxpool);
    network_.push_back(layer2);
    network_.push_back(layer2_activation);
    network_.push_back(layer2_maxpool);
    network_.push_back(layer_flatten);
    network_.push_back(layer3);
    network_.push_back(layer3_activation);

}


void CNN::forward(Eigen::MatrixXf input, Eigen::VectorXf& output, int& y){
    std::vector<Eigen::MatrixXf> x_mat;
    Eigen::VectorXf x_vec;
    x_mat.push_back(input);
    x_mat = network_[0].Conv2d(x_mat);
    x_mat = network_[1].ReLU(x_mat);
    x_mat = network_[2].MaxPool2d(x_mat);
    x_mat = network_[3].Conv2d(x_mat);
    x_mat = network_[4].ReLU(x_mat);
    x_mat = network_[5].MaxPool2d(x_mat);
    x_vec = DLMATH::flatten(x_mat);
    x_vec = network_[6].Linear(x_vec);
    x_vec = network_[7].Softmax(x_vec);
    output_ = x_vec;

    // The index of the maximum value in the output layer is the prediction of the number
    output.maxCoeff(&y);
}

void CNN::forward(std::vector<Eigen::MatrixXf> input, std::vector<Eigen::VectorXf>& output, std::vector<int>& y){
    int T = input.size();  // Number of the batch size
    output.resize(T);
    y.resize(T);
    for(int t=0; t<T; t++){
        forward(input[t], output[t], y[t]);
    }
}