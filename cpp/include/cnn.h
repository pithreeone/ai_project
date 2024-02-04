#include <Eigen/Dense>
#include <string>
#include <vector>

#include <nn.h>

using namespace std;


class CNN{
public:
    
    double loss_;
    bool network_is_legal_;
    vector<NN> network_;

    // Constructor: set the structure of neural network
    CNN();
    
    // Print the network of CNN
    void print();

    // check if network is legal, change the variable: network_is_legal_
    void checkNetwork();

    void forward(Eigen::MatrixXf input, Eigen::VectorXf* output, int* y);

};