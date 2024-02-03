#ifndef _OPTIMIZATION_H_
#define _OPTIMIZATION_H_

#include <vector>
#include <Eigen/Dense>

using namespace std;


class Optimization{
public:
    Optimization();
    
    void calculateLoss(vector<Eigen::MatrixXd> predict, vector<Eigen::MatrixXd> label);
    
    // set all the derivatives to zero
    void zero_grad();

    void backward();

    // fresh all weights that have already pre-calculated
    void step();

};

#endif