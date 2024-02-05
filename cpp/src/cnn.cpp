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

    // Output Layer
    NN output;
    output.Linear(7*7*32, 10);
    output.Softmax();


    // push the above layer to the variable: network
    network_.push_back(layer1);
    network_.push_back(layer1_activation);
    network_.push_back(layer1_maxpool);
    network_.push_back(layer2);
    network_.push_back(layer2_activation);
    network_.push_back(layer2_maxpool);
    network_.push_back(output);

}


void CNN::forward(Eigen::MatrixXf input, Eigen::VectorXf& output, int& y){
    vector<Eigen::MatrixXf> x;
    for(auto i=network_.begin(); i!=network_.end(); i++){
        if(i->getFunctionType() == "Conv2d"){
            x = i->Conv2d(x);
        }
        else if(i->getFunctionType() == "MaxPool2d"){
            x = i->MaxPool2d(x);
        }
        else if(i->getFunctionType() == "Linear"){
            x = i->Linear(x);
        }
        else if(i->getFunctionType() == "ReLU"){
            x = i->ReLU(x);
        }
        else if(i->getFunctionType() == "Sigmoid"){
            x = i->Sigmoid(x);
        }
        else if(i->getFunctionType() == "Softmax"){
            output = DLMATH::flatten(x);
            output = i->Softmax(output);
        }
    }
}