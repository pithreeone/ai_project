#include <Eigen/Dense>
#include <string>
#include <vector>

#include <nn.h>

using namespace std;


class CNN{
public:
    
    double loss_;
    bool network_is_legal_;
    vector<NN> network;

    // Constructor: set the structure of neural network
    CNN();
    
    // pass the data from the input layer to output layer and get the prediction
    void forward(Eigen::MatrixXd input_data);

    // check if network is legal, change the variable: network_is_legal_
    void checkNetwork();

    void optimize();
};