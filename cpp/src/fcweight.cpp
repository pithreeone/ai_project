#include "fcweight.h"

FCWeight::FCWeight(){

}

FCWeight::FCWeight(int in_channels, int out_channels){
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    resize(in_channels_, out_channels_);
}

void FCWeight::resize(int in_channels, int out_channels){
    weights_.resize(in_channels_, out_channels_);
    weights_derivative_.resize(in_channels_, out_channels_);
}

void FCWeight::zero_grad(){
    weights_derivative_.setZero();
}