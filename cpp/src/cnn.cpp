#include "cnn.h"
#include "iostream"

// The class define the neural network structure, 
// using the user-defined class: NN, which is defined in nn.h


CNN::CNN(){
    std::cout << "Initialize network structure of CNN" << std::endl;

    // Hidden Layer 1
    NN layer1, layer1_maxpool;
    layer1.Conv2d(1, 16, 5, 1, 2);
    layer1.ReLU();
    // MaxPool Layer
    layer1_maxpool.MaxPool2d(2);

    // Hidden Layer 2
    NN layer2, layer2_maxpool;
    layer2.Conv2d(16, 32, 5, 1, 2);
    layer2.ReLU();
    // MaxPool Layer
    layer2_maxpool.MaxPool2d(2);

    // Output Layer
    NN output;
    output.Linear(7*7*32, 10);


    // push the above layer to the variable: network
    network_.push_back(layer1);
    network_.push_back(layer1_maxpool);
    network_.push_back(layer2);
    network_.push_back(layer2_maxpool);
    network_.push_back(output);

}

void CNN::forward(Eigen::MatrixXf input, Eigen::VectorXf* output, int* y){

}