#include "optimization.h"


Optimization::Optimization(CNN& cnn, double lr){
    cnn_ = &cnn;
    lr_ = lr;

}

void Optimization::calculateLoss(std::vector<Eigen::VectorXf> predict, std::vector<int> label){
    auto last_layer_it =  cnn_->network_.rbegin();
    if(last_layer_it->getFunctionType() != "LossFunction"){
        throw std::runtime_error("[optimization.cpp]: The last layer is not LossFunction layer ! Something might be wrong");
    }
    loss_ = 0;
    int T = predict.size();
    int n_class = predict[0].size();

    // clear the responsibility of the loss function layer (dE/dy)
    last_layer_it->responsibility_vec_.clear();
    for(int t=0; t<T; t++){
        loss_ += last_layer_it->calculateLoss(predict[t], label[t]);
        Eigen::VectorXf respons(n_class);
        for(int i=0; i<n_class; i++){
            int r;
            if(label[t] == i){
                r = 1;
            }else{
                r = 0;
            }
            double y = predict[t][i];
            respons(i) = (r-y);
        }
        last_layer_it->responsibility_vec_.push_back(respons);
        // std::cout << last_layer_it->responsibility_vec_[t] << std::endl << std::endl; 

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
    int T = cnn_->input_.size();
    for(auto i=cnn_->network_.rbegin(); i!=cnn_->network_.rend(); i++){   // The last layer is the loss function
        // Activation layer & Loss function layer do not need to update the derivatives
        if(i->getFunctionType() == "Activation" || i->getFunctionType() == "LossFunction"){
            continue;
        }

        // Calculate the responsibility & Update derivatives
        if(i->getFunctionType() == "Linear"){
            int input_size = i->weights_linear_->in_channels_;
            int output_size = i->weights_linear_->out_channels_;

            if((i+1)->getFunctionType() != "Activation"){
                throw std::runtime_error("[optimization.cpp]: The next layer of the Linear layer should be an activation layer !");
            }

            // Update the derivatives
            for(int row=0; row<output_size; row++){
                for(int col=0; col<input_size; col++){
                    double derivative = 0;
                    for(int t=0; t<T; t++){
                        double e = (i+2)->responsibility_vec_[t](row);
                        double f_prime = (i+1)->output_vec_[t](row);
                        double z;
                        if((i-1)->getFunctionType() == "Flatten"){
                            z = (col == input_size-1) ? 1 : (i-1)->getCuboidValueFromVector(t, col); // If col==input_size-1, it's the bias
                        }else if((i-1)->getFunctionType() == "Activation"){
                            z = (col == input_size-1) ? 1 : (i-1)->output_vec_[t](col); // If col==input_size-1, it's the bias
                        }
                        
                        derivative += e * f_prime * z;
                    }
                    derivative = -1 * derivative;
                    i->weights_linear_->weights_derivative_(row, col) = derivative;
                }
            }

            // Calculate the responsibility
            for(int t=0; t<T; t++){
                i->responsibility_vec_[t] = i->weights_linear_->weights_.transpose() * (i+1)->output_vec_[t];
            }
        }
        // Update derivatives
        else if(i->getFunctionType() == "Flatten"){


        }
        else if(i->getFunctionType() == "MaxPool2d"){

        }
        else if(i->getFunctionType() == "Conv2d"){

        }

    }
}

void Optimization::step(){
    for (int l = 0; l < cnn_->network_.size(); l++){
        if (cnn_->network_[l].getFunctionType() == "Conv2d"){
            for (int m = 0; m < cnn_->network_[l].weights_kernel_->kernels_.size(); m++){
                std::vector<Eigen::MatrixXf> delta = cnn_->network_[l].weights_kernel_->kernels_[m].kernel_derivative_;
                std::vector<Eigen::MatrixXf> delta_ = delta;
                for (int k = 0; k < delta.size(); k++){
                    for (int i = 0; i < delta[k].rows(); i++){
                        for (int j = 0; j < delta[k].cols(); j++){
                            delta_[k](i,j) = -lr_ * delta[k](i,j);
                            cnn_->network_[l].weights_kernel_->kernels_[m].kernel_[k](i,j) += delta_[k](i,j);
                        }
                    }
                }
            }
        }else if(cnn_->network_[l].getFunctionType() == "Linear"){
            Eigen::MatrixXf delta_ = -lr_ *cnn_->network_[l].weights_linear_->weights_derivative_;
            for (int i = 0; i < delta_.rows(); i++){
                for (int j = 0; j < delta_.cols(); j++){
                    cnn_->network_[l].weights_linear_->weights_(i,j) += delta_(i,j);
                }
            }
        }
    }
}