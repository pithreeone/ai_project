#include "optimization.h"


Optimization::Optimization(CNN& cnn, double lr){
    cnn_ = &cnn;
    lr_ = lr;

}

void Optimization::calculateLoss(std::vector<Eigen::VectorXf> predict, std::vector<int> label){
    loss_ = 0;
    int T = predict.size();
    for(int t=0; t<T; t++){
        loss_ += (cnn_->network_.end() - 1)->calculateLoss(predict[t], label[t]);
    }
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