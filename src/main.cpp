#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;

void printImg(cv::Mat to_print, std::string msg);

void printEnvironmentInfo() {
    std::cout << "------------------------------------\n";
    std::cout << "   Runtime Environment Information   \n";
    std::cout << "------------------------------------\n";
    std::cout << "OpenCV version : " << CV_VERSION << "\n";

#if defined(__clang__)
    std::cout << "Compiler        : Clang " 
              << __clang_major__ << "." << __clang_minor__ << "\n";
#elif defined(__GNUC__)
    std::cout << "Compiler        : GCC " 
              << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n";
#elif defined(_MSC_VER)
    std::cout << "Compiler        : MSVC " << _MSC_VER << "\n";
#else
    std::cout << "Compiler        : Unknown\n";
#endif

    std::cout << "C++ standard    : ";
#if __cplusplus == 199711L
    std::cout << "C++98\n";
#elif __cplusplus == 201103L
    std::cout << "C++11\n";
#elif __cplusplus == 201402L
    std::cout << "C++14\n";
#elif __cplusplus == 201703L
    std::cout << "C++17\n";
#elif __cplusplus == 202002L
    std::cout << "C++20\n";
#elif __cplusplus > 202002L
    std::cout << "C++23 or newer\n";
#else
    std::cout << "Unknown (" << __cplusplus << ")\n";
#endif

    std::cout << "------------------------------------\n\n";
}

// Public paths for images 
const std::string image_1 = "../src/data/frame_1.png";
const std::string image_2 = "../src/data/frame_2.png";

// Error codes
const int image_not_found = -1;

const int default_val = 0;

// Upper and lower bounds for the colors of the cones

// Light Red
const cv::Scalar lower_red1(150, 40, 200);
const cv::Scalar upper_red1(180, 70, 255);

// Dark Red
const cv::Scalar lower_red2(150, 100, 200);
const cv::Scalar upper_red2(180, 255, 255);

// Blue
const cv::Scalar lower_blue(95, 50, 120); // 95 80 120
const cv::Scalar upper_blue(110, 255, 190);

// Yellow
const cv::Scalar lower_yellow(10, 90, 180); //prev 180
const cv::Scalar upper_yellow(22, 170, 255);

// White
const cv::Scalar lower_white(110, 0, 200);  
const cv::Scalar upper_white(180, 25, 255);

// Light Black
const cv::Scalar lower_black1(0, 40, 40); //180 
const cv::Scalar upper_black1(20, 255, 100); //255

// Dark Black
const cv::Scalar lower_black2(160, 40, 50); 
const cv::Scalar upper_black2(180, 70, 70); //195

// Constants for image processing
int bgr2hsv = cv::COLOR_BGR2HSV;

int processing_itrations = 10;
int white_index = 0;
int light_black_index = 1;
int dark_black_index = 2;
int light_red_index = 3;
int make_close = cv::MORPH_CLOSE;
int make_open = cv::MORPH_OPEN;
cv::Mat strong_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)); // best one so far
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)); // best one so far

int main() {

    printEnvironmentInfo();

    // Level 1: load and display the image 
    cv::Mat frame_1 = cv::imread(image_1);
    if (frame_1.empty()) {
        std::cerr << "Image not found!\n";
        return image_not_found;
    }

    printImg(frame_1,"Frame 1 clear");
    
    // Level 2: Cone detection

    // Pre processing of the image:

    // Changing color mode from bgr to hsv in order to remove color brightness and saturation  

    cv::Mat hsv_frame_1;
    cv::cvtColor(frame_1, hsv_frame_1, cv::COLOR_BGR2HSV);

    // Struct to memoriza the bounds of the colors and the images at every step of the pipeline
    std::vector <std::string> color_names ={ // String names of the colors
        "white",
        "light black",
        "dark black",
        "dark_red",
        "blue",
        "yellow",
    };
    std::vector <std::pair<cv::Scalar, cv::Scalar>> color_bounds = { // pairs of the lower and upper bound for each color
        {lower_white, upper_white},
        {lower_black1, upper_black1},
        {lower_black2, upper_black2},
        {lower_red2, upper_red2},
        {lower_blue, upper_blue},
        {lower_yellow, upper_yellow},
    };
    std::vector<cv::Mat> frames_cones_color(color_bounds.size()); // color detector for each color of the cones
    std::vector<std::vector<std::vector<cv::Point>>> colors_contours(color_bounds.size()); // contours of each color

    // Pipeline application for each image of the cones color

    for(size_t i = 0; i < color_bounds.size(); i++)
    {
        const std::pair<cv::Scalar,cv::Scalar> bounds = color_bounds[i]; //bounds of the colors of each cones

        // Extract the current color
        cv::inRange(hsv_frame_1, bounds.first, bounds.second, frames_cones_color[i]); 
        
        // Tune the current mask with another one in order to combine different colors to create the best profile for the cones
        if(i == 2)
        {
            cv::bitwise_or(frames_cones_color[i], frames_cones_color[1], frames_cones_color[i]);
        }
        if (i == 3)
        {
            cv::bitwise_or(frames_cones_color[i], frames_cones_color[white_index], frames_cones_color[i]);
            cv::dilate(frames_cones_color[i], frames_cones_color[i], strong_kernel);
        }
        if(i == 4)
        {
            cv::bitwise_or(frames_cones_color[i], frames_cones_color[white_index], frames_cones_color[i]);
            cv::dilate(frames_cones_color[i], frames_cones_color[i], strong_kernel);
        }
        
        if(i == 5)
        {
            cv::bitwise_or(frames_cones_color[i], frames_cones_color[dark_black_index], frames_cones_color[i]);
            cv::dilate(frames_cones_color[i], frames_cones_color[i], strong_kernel);
        }
        // Join adjacent areas
        cv::morphologyEx(frames_cones_color[i], frames_cones_color[i], make_close, strong_kernel);
        cv::morphologyEx(frames_cones_color[i], frames_cones_color[i], make_open, kernel);

        // Fill black areas inside contours using another mask
        cv::Mat im_floodfill = frames_cones_color[i].clone();
        cv::floodFill(im_floodfill, cv::Point(0, 0), cv::Scalar(255)); // riempi da bordo con bianco

        cv::bitwise_not(im_floodfill, im_floodfill);

        // Fill the holes inside the previous image
        frames_cones_color[i] = (frames_cones_color[i] | im_floodfill);

        // Hierarchy declaration for contours and copying the mask to further improve the pipeline
        std::vector<cv::Vec4i> hierarchy;
        cv::Mat mask_copy = frames_cones_color[i].clone();

        // Find the contours of the white areas
        cv::findContours(frames_cones_color[i], colors_contours[i], hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if(i > dark_black_index)
        {
            //printImg(frames_cones_color[i], color_names[i]); 
            for (const auto& contour : colors_contours[i]) {
                std::vector<cv::Point> approx;
                double epsilon = 0.02 * cv::arcLength(contour, true); 
                cv::approxPolyDP(contour, approx, epsilon, true);

                if (approx.size() >= 3 && approx.size() <= 9) {
                    // Check if the shape is similar to a triangle
                    double area = cv::contourArea(contour);
                    cv::Rect box = cv::boundingRect(contour);
                    double aspectRatio = (double)box.width / box.height;

                    if(area > 15 && aspectRatio <= 5)
                    {
                        cv::Mat roi = hsv_frame_1(box); // ritagli la regione nel frame HSV

                        // creating a mask for each color
                        cv::Mat mask_blue, mask_yellow, mask_red, mask_white;

                        cv::inRange(roi, lower_blue, upper_blue, mask_blue);
                        cv::inRange(roi, lower_yellow, upper_yellow, mask_yellow);
                        cv::inRange(roi, lower_red2, upper_red2, mask_red);
                        cv::inRange(roi, lower_white, upper_white, mask_white);

                        int contour_area = (box.width * box.height);

                        // Calculating how many of the pixel with the same color as the cones are in each boc
                        double blue_ratio = (double)cv::countNonZero(mask_blue) / contour_area;
                        double yellow_ratio = (double)cv::countNonZero(mask_yellow) / contour_area;
                        double red_ratio = (double)cv::countNonZero(mask_red) / contour_area;
                        double white_ratio = (double)cv::countNonZero(mask_white) / contour_area;

                        // Level 3: classifying each cones with a different borders color based on the ones of the found one
                        // Draw only those boxes with the colors of the cones and a relatively small amount of white
                        //  This is to prevent openCV from detecting the white stripe of a red or blue cone as a separate item
                        if ( (blue_ratio > 0.1 || yellow_ratio > 0.1 || red_ratio > 0.1) && white_ratio < 0.1 ) {
                            cv::rectangle(frame_1, box, bounds.first, 2);
                            //printImg(frame_1,"cones detected so far");
                        }
                    }
                }
            }
        }
    }
    printImg(frame_1,"Cones detected");
    return default_val;
}

void printImg(cv::Mat to_print, std::string msg)
{
    cv::imshow(msg, to_print);
    cv::waitKey(default_val);
}