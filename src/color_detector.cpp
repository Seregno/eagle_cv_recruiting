#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void onMouse(int event, int x, int y, int, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Mat* hsv = reinterpret_cast<cv::Mat*>(userdata);
        cv::Vec3b pixel = hsv->at<cv::Vec3b>(y, x);
        std::cout << "H: " << (int)pixel[0] << " S: " << (int)pixel[1] << " V: " << (int)pixel[2] << std::endl;
    }
}

int main() {
    cv::Mat image = cv::imread("../src/data/frame_1.png");
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    cv::imshow("Image", image);
    cv::setMouseCallback("Image", onMouse, &hsv);
    cv::waitKey(0);
}