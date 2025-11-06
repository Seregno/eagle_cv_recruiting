#include "constants.h"

// Public paths for images 
const std::string image_1 = "../src/data/frame_1.png";
const std::string image_2 = "../src/data/frame_2.png";

const int image_not_found = -1; // value for a missing image
const int default_val = 0; // default value for a good operation

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
const cv::Scalar lower_white(110, 0, 200); //(110, 0, 200)  
const cv::Scalar upper_white(180, 35, 255); // (180, 25, 255)

// Light Black
const cv::Scalar lower_black1(0, 40, 40); //180 
const cv::Scalar upper_black1(20, 255, 100); //255

// Dark Black
const cv::Scalar lower_black2(160, 40, 50); 
const cv::Scalar upper_black2(180, 70, 70); //195

// Vector to memorize the names of the colors of the cones
const std::vector <std::string> color_names ={ // String names of the colors
    "white",
    "light black",
    "dark black",
    "dark_red",
    "blue",
    "yellow",
};

// Vector whose purpose is to memorize the upper and lower bound fo each color
const std::vector <std::pair<cv::Scalar, cv::Scalar>> color_bounds = { 
    {lower_white, upper_white},
    {lower_black1, upper_black1},
    {lower_black2, upper_black2},
    {lower_red2, upper_red2},
    {lower_blue, upper_blue},
    {lower_yellow, upper_yellow},
};

const double main_color_bound = 0.1; // minimum percentage for the main color of a cone
const double secondary_color_bound = 0.1; // maximum percentage for the secondary color of a cone
const int point_for_orb = 2000; // Number of points that orb will look for
const int left_red_cone_selected = 0; // Mode value for selecting the leftomost red cone during cone detection
const int right_red_cone_selected = 1; // Mode value for selecting the rightomost red cone during cone detection
/*
// Indexes of processing order for the colors
const int white_index = 0; 
const int light_black_index = 1;
const int dark_black_index = 2;
const int red_index = 3;
const int blue_index = 4;
const int yellow_index = 5;
*/

// kernels to use for image processing
const cv::Mat kernel_9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
const cv::Mat kernel_7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)); 
const cv::Mat kernel_5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
const cv::Mat kernel_3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); 

// Unvalid point for cone baricenter initialization
const cv::Point invalid_point = cv::Point(-1,-1);

// Essential matrix for Level 5

cv::Mat K = (cv::Mat_<double>(3, 3) <<
    387.3502807617188, 0,                 317.7719116210938,
    0,                 387.3502807617188, 242.4875946044922,
    0,                 0,                 1);