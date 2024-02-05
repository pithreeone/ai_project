#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <Eigen/Dense>
#include <vector>

// The class define the 3d rectangle of the kernel and a bias
class Kernel3d{
public:
    int kernel_size_;
    int channels_;
    std::vector<Eigen::MatrixXf> kernel_;
    std::vector<Eigen::MatrixXf> kernel_derivative_;
    double bias_;
    double bias_derivative_;

    Kernel3d();

    Kernel3d(int kernel_size, int channels);

    // initialize weights randomly with minimum and maximum value
    void randomIntialize(double min, double max);

    // print the information of kernel: size, number of weights
    void print();
};

#endif