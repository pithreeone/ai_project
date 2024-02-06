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

    cout << "hello_world" << endl;
    // load data
    // DataLoader dl(data_path_root);
    // dl.loadDataFromFolder();
    // dl.getData(train_x, train_y, test_x, test_y);
    
    CNN cnn;
    
    // Below is test-code
    Eigen::MatrixXf input_temp(3, 5);
    input_temp << 1.1, 2, 3, 4, 5,
                4, 5, 6, 7, 8,
                9, 10, 11, 12, -1; // Ensure you're providing all necessary elements.
    std::vector<Eigen::MatrixXf> input, output;
    input.push_back(input_temp);
    std::cout << input[0] << std::endl;
    output = cnn.network_[2].MaxPool2d(input);
    std::cout << output[0] << std::endl;
    // Above is test-code

    int n_train;
    Optimization optimization(cnn, LR);

    // Set the loss function
    NN loss_func;
    loss_func.CrossEntropyLoss();
    cnn.network_.push_back(loss_func);

    // Iterate complete pass through the entire training dataset
    for(int epoch = 0; epoch < EPOCH; epoch++){

        // Iterate an entire dataset and divide in BATCH_SIZE
        int step_max = ceil(n_train / BATCH_SIZE);
        for(int step; step < step_max; step++){

            // Get the data of this batch
            std::vector<Eigen::MatrixXf> batch_x;
            std::vector<int> batch_y;

            // Do the forward propagation (save all the results in each layer)
            std::vector<Eigen::VectorXf> output;
            std::vector<int> y;
            cnn.forward(batch_x, output, y);

            // calculate loss (save the result in optimization.loss_)
            optimization.calculateLoss(output, batch_y);

            // set all the gradients to zero
            optimization.zero_grad();

            // calculate all the derivatives
            optimization.backward();

            // update the weights by the pre-calculated derivatives
            optimization.step();
            
            if (step == step_max - 1){
                // calculate the loss and accuracy in this epoch
            }




        }
    }

    return 0;
}
