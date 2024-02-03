#include "cnn.h"
#include "iostream"
CNN::CNN(){
    std::cout << "Initialize CNN object" << std::endl;

    // Hidden Layer 1
    NN layer1;
    layer1.Conv2d(1, 16, 5, 1, 2);
    layer1.ReLU();
    layer1.MaxPool2d(2);

    // Hidden Layer 2
    NN layer2;
    layer2.Conv2d(16, 32, 5, 1, 2);
    layer2.ReLU();
    layer2.MaxPool2d(2);

    // Output Layer
    NN output;
    output.Linear(32*7*7, 10);


    // push the above layer to the variable: network
    network.push_back(layer1);
    network.push_back(layer2);
    network.push_back(output);

}