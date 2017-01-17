#include<iostream>

#include<opencv2/opencv.hpp>
#include<image_transport/image_transport.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<sensor_msgs/Image.h>
#include<cv_bridge/cv_bridge.h>
#include<signal.h>
#include<string.h>

cv::VideoCapture cam;
cv::Mat frame;

void mouseAction(int event, int x, int y, int flags, void* ){

    if(event == CV_EVENT_LBUTTONDOWN){
        std::stringstream filename;
        filename<<"/home/isl-server/py-faster-rcnn/data/demo/click.jpg";
        cv::imwrite( filename.str(), frame );
        std::cout<<"image has been clicked.\n";
    }

}

int main(int argc, char** argv)
{
    ros::init( argc, argv, "snapshot");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    image_transport::Publisher pub=it.advertise("logitech/webcam_raw", 2);

    cam.open(0);
    cam.set(3, 640);
    cam.set(4, 480);

    sensor_msgs::ImagePtr msg;

    while (ros::ok())
    {
        cam >> frame;

        if(!frame.empty())
        {
            cv::imshow("webcam", frame);
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
            pub.publish(msg);

            cv::setMouseCallback("webcam", mouseAction, NULL);
            cv::waitKey(1);
        }
    }

    ros::spinOnce();
    return 0;

}





