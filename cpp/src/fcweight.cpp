#include "fcweight.h"

FCWeight::FCWeight(){

}

FCWeight::FCWeight(int in_channels, int out_channels){
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    resize(out_channels_, in_channels_ + 1);
}

void FCWeight::resize(int in_channels, int out_channels){
    weights_.resize(out_channels_, in_channels_ + 1);
    weights_derivative_.resize(out_channels_, in_channels_ + 1);
}

void FCWeight::zero_grad(){
    weights_derivative_.setZero();
}