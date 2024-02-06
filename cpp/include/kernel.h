#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <Eigen/Dense>
#include <vector>
#include <iostream>

// The class define the 3d rectangle of the kernel and a bias
class Kernel3d{
public:
    int kernel_size_;
    int channels_;
    std::vector<Eigen::MatrixXf> kernel_;               // weights
    std::vector<Eigen::MatrixXf> kernel_derivative_;    // derivatives
    double bias_;
    double bias_derivative_;

    Kernel3d();

    Kernel3d(int kernel_size, int channels);

    // initialize weights randomly with minimum and maximum value
    void randomIntialize(double min, double max);

    // overload operator "<<"
    friend std::ostream& operator<<(std::ostream& os, const Kernel3d& obj);
};

// Handle the weights of the 4d kernel which is just the multiple Kernel3d.
class Kernel4d{
public:
    int kernel_size_;
    int in_channels_;
    int out_channels_;
    std::vector<Kernel3d> kernels_;

    Kernel4d();
    Kernel4d(int kernel_size, int in_channels, int out_channels);

    // Set all the derivatives in kernels_ to zero.
    void zero_grad();

    // overload operator "<<"
    friend std::ostream& operator<<(std::ostream& os, const Kernel4d& obj);
};

#endif