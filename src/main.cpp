#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void printEnvironmentInfo();
void printImg(cv::Mat to_print, std::string msg);
double point_distance(const cv::Point& a, const cv::Point& b);
cv::Point median_point(const cv::Point& a, const cv::Point& b);
void assign_red_cone(cv::Point& red_cone, const cv::Point& new_cone, const int selected_cone);
cv::Point get_circuit_point(const cv::Point& current_cone, std::vector <cv::Point>& other_side_cones);
void sortCircuitPoints(const cv::Point& starting_point,std::vector<cv::Point>& circuit_points);
void drawCircuit(cv::Mat& image, const std::vector<cv::Point>& circuit_points);

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

const int left_red_cone_selected = 0;
const int right_red_cone_selected = 1;
const int processing_itrations = 10;
const int white_index = 0;
const int light_black_index = 1;
const int dark_black_index = 2;
const int red_index = 3;
const int blue_index = 4;
const int yellow_index = 5;
cv::Mat strong_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)); // best one so far
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)); // best one so far
cv::Point invalid_point = cv::Point(-1,-1);
cv::Point leftmost_red_cone = cv::Point(-1,-1);
cv::Point rightmost_red_cone = cv::Point(-1,-1);
std::vector <cv::Point> blue_cones;
std::vector <cv::Point> yellow_cones;
std::vector <cv::Point> circuit_points;

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
        if( i > 1 )
        {
            switch (i)
            {
                case dark_black_index:
                    cv::bitwise_or(frames_cones_color[i], frames_cones_color[1], frames_cones_color[i]);
                break;

                case red_index:
                    cv::bitwise_or(frames_cones_color[i], frames_cones_color[white_index], frames_cones_color[i]);
                    cv::dilate(frames_cones_color[i], frames_cones_color[i], strong_kernel);
                break;

                case blue_index:
                    cv::bitwise_or(frames_cones_color[i], frames_cones_color[white_index], frames_cones_color[i]);
                    cv::dilate(frames_cones_color[i], frames_cones_color[i], strong_kernel);
                break;

                case yellow_index:
                    cv::bitwise_or(frames_cones_color[i], frames_cones_color[dark_black_index], frames_cones_color[i]);
                    cv::dilate(frames_cones_color[i], frames_cones_color[i], strong_kernel);
                break;

                default:
                    cout <<"No color tuning needed" <<endl;
                break;
            }
        }

        // Join adjacent areas
        cv::morphologyEx(frames_cones_color[i], frames_cones_color[i], cv::MORPH_CLOSE, strong_kernel);
        cv::morphologyEx(frames_cones_color[i], frames_cones_color[i], cv::MORPH_OPEN, kernel);

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
                            cv::Point new_cone = cv::Point( box.x + (box.width / 2), box.y + (box.height / 2) ); // creating the coordinates of the new cones corresponding to the ones of the center of the already drawn box
                            
                            // Storing and classifying the new cone in order to use its coordinates to detect the circuit in the next level
                            switch (i)
                            {
                                case red_index: // checking if the new red cone is the left most or right most
                                    assign_red_cone(leftmost_red_cone, new_cone ,left_red_cone_selected);
                                    assign_red_cone(rightmost_red_cone, new_cone ,right_red_cone_selected);
                                break;
                                
                                case blue_index:
                                    blue_cones.push_back(new_cone);
                                break;
                                
                                case yellow_index:
                                    yellow_cones.push_back(new_cone);
                                break;
                                
                                default:
                                    cout <<"Unknown item detected" <<endl;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    printImg(frame_1,"Cones detected");

    // Level 4: detect circuit

    // Detect the point of the circuit by calculating the median point between the current blue/yellow cone with the closest one on the other side

    cv::Point starting_point = median_point(leftmost_red_cone, rightmost_red_cone);
    circuit_points.push_back(starting_point);
    for(size_t i = 0; i < blue_cones.size(); i++)
    {
        cv::Point new_circuit_point = get_circuit_point(blue_cones[i], yellow_cones);
        circuit_points.push_back(new_circuit_point);
    }

    for(size_t i = 0; i < yellow_cones.size(); i++)
    {
        cv::Point new_circuit_point = get_circuit_point(yellow_cones[i], blue_cones);
        circuit_points.push_back(new_circuit_point);
    }
    
    // Sort the points of the circuit based on how close they are to a starting point

    sortCircuitPoints(starting_point, circuit_points);
    drawCircuit(frame_1,circuit_points);
    printImg(frame_1, "Circuit detected");

    return default_val;
}

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

void printImg(cv::Mat to_print, std::string msg)
{
    cv::imshow(msg, to_print);
    cv::waitKey(default_val);
}

double point_distance(const cv::Point& a, const cv::Point& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

cv::Point median_point(const cv::Point& a, const cv::Point& b) {
    return cv::Point((a.x + b.x) / 2, (a.y + b.y) / 2);
}

void assign_red_cone(cv::Point& red_cone, const cv::Point& new_cone, const int selected_cone) {
    if( selected_cone == left_red_cone_selected && ( (red_cone.x == invalid_point.x && red_cone.y == invalid_point.y) || (red_cone.x > new_cone.x) ) ) // unassigned cone
    {
        red_cone.x = new_cone.x;
        red_cone.y = new_cone.y;
    }
    else if( selected_cone == right_red_cone_selected && ( (red_cone.x == invalid_point.x && red_cone.y == invalid_point.y) || (red_cone.x < new_cone.x) ) )
    {
        red_cone.x = new_cone.x;
        red_cone.y = new_cone.y;
    }   
}

cv::Point get_circuit_point(const cv::Point& current_cone, std::vector <cv::Point>& other_side_cones)
{
    if (other_side_cones.empty())
    {
        return current_cone;
    }
    cv::Point closest_cone = other_side_cones[0];
    double best_len = point_distance(current_cone, other_side_cones[0]);
    double current_len;
    for(size_t i = 1; i < other_side_cones.size(); i++)
    {
        current_len = point_distance(current_cone, other_side_cones[i]);
        if (current_len < best_len)
        {
            closest_cone = other_side_cones[i];
            best_len = current_len;
        }
    }
    return median_point(current_cone, closest_cone);
}

// Sort the points of the circuit considering first the ones with a bigger x value, eventually sorting considering  y value

void sortCircuitPoints(const cv::Point& starting_point,std::vector<cv::Point>& circuit_points) {
    std::sort(circuit_points.begin(), circuit_points.end(),
              [&starting_point](const cv::Point& a, const cv::Point& b) {
                  return point_distance(a,starting_point) > point_distance(b, starting_point);          
              });
}

void drawCircuit(cv::Mat& image, const std::vector<cv::Point>& circuit_points) {
    // Drawing every point as a circle
    for (const auto& pt : circuit_points) {
        cv::circle(image, pt, 5, cv::Scalar(0, 0, 255), -1); // rosso, riempito
    }

    // Joining points with lines
    for (size_t i = 1; i < circuit_points.size(); ++i) {
        cv::line(image, circuit_points[i-1], circuit_points[i], cv::Scalar(255, 0, 0), 2); // blu, spessore 2
    }
}
