#include "optimization.h"


Optimization::Optimization(CNN& cnn, double lr){
    cnn_ = &cnn;
    lr_ = lr;

}

void Optimization::calculateLoss(std::vector<Eigen::VectorXf> predict, std::vector<int> label){

}

void Optimization::zero_grad(){
    // Iterate all the layer of the network.
    for(auto i=cnn_->network_.begin(); i!=cnn_->network_.end(); i++){
        i->weights_kernel_->zero_grad();
        i->weights_linear_->zero_grad();
    }
}

void Optimization::backward(){

}

void Optimization::step(){

}