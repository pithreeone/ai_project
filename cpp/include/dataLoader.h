#ifndef DATALOADER_H
#define DATALOADER_H

using namespace std;
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>


class DataLoader{
public:
    string data_path_root_;

    vector<Eigen::MatrixXd> train_x_, test_x_;
    vector<int> train_y_, test_y_;

    // Constructor: initialize some parameters... 
    DataLoader(string data_path_root);

    // use opencv library to load data
    void loadDataFromFolder();

    // randomize the data
    void shuffleData();

    // &train_x, &train_y, &test_x, &test_y
    void getData(vector<Eigen::MatrixXd>& train_x, vector<int>& train_y, 
                vector<Eigen::MatrixXd>& test_x, vector<int>& test_y);

    void printMetaData();
};


#endif