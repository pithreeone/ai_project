#include "dataLoader.h"
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;

DataLoader::DataLoader(string data_path_root){
    data_path_root_ = data_path_root;
}

void DataLoader::loadDataFromFolder(){
    string folderPath = data_path_root_ + "/train/0";
    // Iterate over all the files in the directory
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
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
                cv::waitKey(0); // Wait for a key press
                Eigen::MatrixXd eigenImage(image.rows, image.cols);
                for(int i = 0; i < image.rows; ++i) {
                    for(int j = 0; j < image.cols; ++j) {
                        eigenImage(i, j) = image.at<uchar>(i, j);
                    }
                }
                train_x_.push_back(eigenImage);
                train_y_.push_back();
            } else {
                std::cerr << "Failed to read image: " << filePath << std::endl;
            }
        }
    }
}