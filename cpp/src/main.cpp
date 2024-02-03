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

using namespace std;


// Parameters
string data_path_root = "/root/ai_project/cpp/data";

int EPOCH = 3;
int BATCH_SIZE = 50;
double LR = 0.001;


int main(int argc, char** argv){

    cout << "hello_world" << endl;
    DataLoader dl(data_path_root);

    Optimization optimization;
    NN loss_func;
    loss_func.CrossEntropyLoss();

    for(int epoch = 0; epoch < EPOCH; epoch++){
        int STEP;
        for(int step; step < STEP; step++){
            vector<Eigen::MatrixXd> batch_x, batch_y;
            vector<Eigen::MatrixXd> output;
            optimization.calculateLoss(output, batch_y);
            optimization.zero_grad();
            // calculate all the derivatives
            optimization.backward();
            // fresh the weights by the pre-calculated derivatives
            optimization.step();
            
            if (step == STEP - 1){
                // calculate the loss and accuracy in this epoch
            }

        }
    }


    CNN cnn;
    return 0;
}
