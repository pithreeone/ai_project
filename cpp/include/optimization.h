#ifndef _OPTIMIZATION_H_
#define _OPTIMIZATION_H_

#include <vector>
#include <Eigen/Dense>
#include "nn.h"

// Update the parameters, 
// when initialize the object, need to parse the parameters that need to update
class Optimization{
private:
    double lr_;
    vector<NN> network_;
public:
    Optimization(vector<NN>& network);
    
    void calculateLoss(vector<Eigen::MatrixXd> predict, vector<Eigen::MatrixXd> label);
    
    // set all the derivatives to zero
    void zero_grad();

    // pass the data from the input layer to output layer, calculate the prediction and the value of hidden layer
    void forward(Eigen::MatrixXd input_data);

    // calculate the derivatives from last layer to input layer
    void backward();

    // fresh all weights that have already pre-calculated
    void step();

};

#endif