#ifndef _OPTIMIZATION_H_
#define _OPTIMIZATION_H_

#include <vector>
#include <Eigen/Dense>
#include "cnn.h"

// The function of the class is doing the backpropogation and do the gradient descent

class Optimization{
private:
    double loss_;
    double lr_;
    CNN* cnn_;

public:

    // when initialize the object, you need to parse the parameters that need to update
    Optimization(CNN& cnn, double lr);
    
    double getLoss(){ return loss_; }

    // Calculate loss and save in variable:loss_
    // The input argument are vectors of prediction and label. The length of the vector is BATCH_SIZE.
    void calculateLoss(std::vector<Eigen::VectorXf> predict, std::vector<int> label);
    
    // set all the derivatives to zero
    void zero_grad();

    // pass the data from the input layer to output layer, calculate the prediction and the value of hidden layer
    void forward(Eigen::MatrixXd input_data);

    // calculate the derivatives from last layer to input layer
    void backward();

    // Update all weights. The derivatives should have already been pre-calculated
    void step();

};

#endif