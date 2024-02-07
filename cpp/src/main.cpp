/**
 * Main program for a machine learning project using Convolutional Neural Networks (CNN).
 * 
 * This program demonstrates the setup and basic usage of a CNN model for an AI project in C++. 
 * It includes essential libraries such as Eigen for numerical operations, and custom headers 
 * like cnn.h and dataLoader.h for the CNN model and data handling. The program initializes 
 * key parameters for the training process, such as the number of epochs, batch size, and learning 
 * rate. It then creates an instance of the CNN model and can be extended to perform training and 
 * evaluation tasks on a specified dataset.
 *
 * Authors:
 *     Fiona - [Additional information about Fiona, like her role or contact information]
 *     Ben - [Additional information about Ben, like his role or contact information]
 *
 * Usage:
 *     Compile and run the program. The main function initializes the model and may be extended 
 *     to include data loading, model training, and evaluation phases.
 */

#include <iostream>
#include <nn.h>
#include <cnn.h>
#include <dataLoader.h>
#include <optimization.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <dlmath.h>
#include <cmath>


using namespace std;


// Parameters
string data_path_root = "/root/ai_project/cpp/data";

int EPOCH = 3;
int BATCH_SIZE = 50;
double LR = 0.001;
vector<Eigen::MatrixXd> train_x, test_x;
vector<int> train_y, test_y;

int main(int argc, char** argv){

    // load data
    // DataLoader dl(data_path_root);
    // dl.loadDataFromFolder();
    // dl.getData(train_x, train_y, test_x, test_y);
    
    CNN cnn;
    
    // Below is test-code
    // Kernel3d kernel(5, 2);
    // Eigen::MatrixXf k(5, 5);
    // k << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24;
    // kernel.kernel_[0] = kernel.kernel_[1] = k;
    // Kernel4d kernels(5, 2, 3);
    // kernels.kernels_[0] = kernels.kernels_[1] = kernels.kernels_[2] = kernel;
    // std::cout << kernels << std::endl;;
    // Above is test-code


    // // Set the loss function
    NN loss_func;
    loss_func.CrossEntropyLoss();
    cnn.network_.push_back(loss_func);

    // Below is test-code
    // int n_train;
    // Optimization optimization(cnn, LR);
    // std::vector<Eigen::VectorXf> prediction;
    // Eigen::VectorXf temp(5);
    // temp << 0.3, 0.4, 0.1, 0.1, 0.1;
    // prediction.push_back(temp);
    // prediction.push_back(temp);
    // std::vector<int> label;
    // label.push_back(1);
    // label.push_back(0);
    // optimization.calculateLoss(prediction, label);
    // std::cout << optimization.getLoss() << std::endl;
    // Above is test-code

    // Below is test-code
    // NN layer1, layer1_activation, layer1_maxpool;
    // layer1.Conv2d(1, 16, 5, 1, 2);
    // cout << layer1.output_mat_ << endl;
    // layer1_activation.ReLU();
    // cout << layer1_activation.output_mat_ << endl;
    // layer1_maxpool.MaxPool2d(2);
    // cout << layer1_maxpool.output_mat_ << endl;
    // NN layer3, layer3_activation;
    // layer3.Linear(14*14*16, 10);
    // layer3_activation.Softmax();
    // Above is test-code

    // Below is test-code
    // Eigen::MatrixXf weight(2,3);
    // Eigen::VectorXf x(3);
    // weight << 1,2,3,4,5,6;
    // x << 1,1,1;
    // cout << weight*x << endl;
    // Above is test-code

    // // Iterate complete pass through the entire training dataset
    // for(int epoch = 0; epoch < EPOCH; epoch++){

    //     // Iterate an entire dataset and divide in BATCH_SIZE
    //     int step_max = ceil(n_train / BATCH_SIZE);
    //     for(int step; step < step_max; step++){

    //         // Get the data of this batch
    //         std::vector<Eigen::MatrixXf> batch_x;
    //         std::vector<int> batch_y;

    //         // Do the forward propagation (save all the results in each layer)
    //         std::vector<Eigen::VectorXf> output;
    //         std::vector<int> y;
    //         cnn.forward(batch_x, output, y);

    //         // calculate loss (save the result in optimization.loss_)
    //         optimization.calculateLoss(output, batch_y);

    //         // set all the gradients to zero
    //         optimization.zero_grad();

    //         // calculate all the derivatives
    //         optimization.backward();

    //         // update the weights by the pre-calculated derivatives
    //         optimization.step();
            
    //         if (step == step_max - 1){
    //             // calculate the loss and accuracy in this epoch
    //         }




    //     }
    // }

    return 0;
}
