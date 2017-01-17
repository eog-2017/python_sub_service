#include<iostream>

#include<opencv2/opencv.hpp>
#include<image_transport/image_transport.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<sensor_msgs/Image.h>
#include<cv_bridge/cv_bridge.h>
#include<signal.h>
#include<string.h>

void imageCaptureCallback(const sensor_msgs::Image::ConstPtr image){

    std::cout<<"function loop \n";
    cv_bridge::CvImagePtr bridge;
    bridge=cv_bridge::toCvCopy(image,"bgr8");

    cv::imshow("sub", bridge->image);
    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    ros::init( argc, argv, "img_sub");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image>("logitech/webcam_raw", 2, imageCaptureCallback);

    ros::Rate loop_rate(20);
        while(ros::ok())
        {
            std::cout<<"while loop \n";
            ros::spin();
            loop_rate.sleep();
        }

        return 0;
}





