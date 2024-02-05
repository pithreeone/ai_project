#include "dataLoader.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

DataLoader::DataLoader(string data_path_root){
    data_path_root_ = data_path_root;
}

void DataLoader::loadDataFromFolder(){
    for (int k = 0; k <10; k++){
        string num_s = to_string(k);
        string folderPath_train = data_path_root_ + "/train/" + num_s;
        string folderPath_test = data_path_root_ + "/test/" + num_s;
        // Iterate over all the files in the directory
        for (const auto& entry : std::filesystem::directory_iterator(folderPath_train)) {
            // Check if the entry is a file and not a directory
            if (entry.is_regular_file()) {
                // Get the path of the file
                std::string filePath = entry.path().string();
                // Read the image
                cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
                if (!image.empty()) {
                    // Successfully read the image, now you can process it
                    // For example, display the image
                    // cv::imshow("Image", image);
                    // cv::waitKey(0); // Wait for a key press
                    Eigen::MatrixXd eigenImage(image.rows, image.cols);
                    for(int i = 0; i < image.rows; ++i) {
                        for(int j = 0; j < image.cols; ++j) {
                            eigenImage(i, j) = image.at<uchar>(i, j);
                        }
                    }
                    train_x_.push_back(eigenImage);
                    train_y_.push_back(k);
                } else {
                    std::cerr << "Failed to read image: " << filePath << std::endl;
                }
            }
        }

        // Iterate over all the files in the directory
        for (const auto& entry : std::filesystem::directory_iterator(folderPath_test)) {
            // Check if the entry is a file and not a directory
            if (entry.is_regular_file()) {
                // Get the path of the file
                std::string filePath = entry.path().string();
                // Read the image
                cv::Mat image = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
                if (!image.empty()) {
                    // Successfully read the image, now you can process it
                    // For example, display the image
                    // cv::imshow("Image", image);
                    // cv::waitKey(0); // Wait for a key press
                    Eigen::MatrixXd eigenImage(image.rows, image.cols);
                    for(int i = 0; i < image.rows; ++i) {
                        for(int j = 0; j < image.cols; ++j) {
                            eigenImage(i, j) = image.at<uchar>(i, j);
                        }
                    }
                    test_x_.push_back(eigenImage);
                    test_y_.push_back(k);
                } else {
                    std::cerr << "Failed to read image: " << filePath << std::endl;
                }
            }
        }
    }
    // cout << train_x_.at(40000) << endl;
    // cout << train_y_.at(40000) << endl;
    // cout << test_x_.at(6000) << endl;
    // cout << test_y_.at(6000) << endl;
}

void DataLoader::getData(vector<Eigen::MatrixXd>& train_x, vector<int>& train_y, vector<Eigen::MatrixXd>& test_x, vector<int>& test_y){
    train_x = train_x_;
    train_y = train_y_;
    test_x = test_x_;
    test_y = test_y_;
}

void DataLoader::printMetaData(){
    
}