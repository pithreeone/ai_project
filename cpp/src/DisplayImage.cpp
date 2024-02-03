#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load an image from file
    cv::Mat img = cv::imread("/root/ai_project/cpp/data/test/0/3.png", cv::IMREAD_COLOR);
    if(img.empty()) {
        std::cout << "Error: Image cannot be loaded." << std::endl;
        return -1;
    }

    // Create a window for display
    cv::namedWindow("Image Window", cv::WINDOW_AUTOSIZE);

    // Show our image inside the created window
    cv::imshow("Image Window", img);

    // Wait for a keystroke in the window
    cv::waitKey(0);

    return 0;
}