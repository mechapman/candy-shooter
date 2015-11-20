//  Open Mouth Detection
//
//  Created by Michael Chapman on 10/31/15.
//  Copyright © 2015 Michael Chapman. All rights reserved.
//

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//Function Headers
void detectAndDisplay( Mat frame );

// Global variables
int threshold_value = 85;
Mat mouthROI;

String face_cascade_name = "/Users/mchapman/Downloads/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
String window_name = "Capture";

int main( void )
{
    VideoCapture capture;
    Mat frame;

    //Load the face cascade
    if( !face_cascade.load( face_cascade_name ) ){
        printf("--(!)Error loading face cascade\n");
        return -1;
    };
    
    //Read the video stream
    capture.open( 0 );
    printf("opened webcam\n");

    if ( ! capture.isOpened() ) {
        printf("--(!)Error opening video capture\n");
        return -1;
    }

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        //-- 3. Apply the classifier, detect open mouths
        detectAndDisplay( frame );
        int c = waitKey(10);
        if( (char)c == 27 ) { break; } // escape

    }
    return 0;
}

void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    
    //convert to grayscale, equalize distribution of pixels
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(100, 100) );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        //draw ellipse around face
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        
        Mat faceROI = frame_gray( faces[i] );
        
        //ROI where mouth will be
        Rect mouth_rect = Rect(faces[i].x + (0.35*faces[i].width), faces[i].y + (0.75*faces[i].height), (0.35*faces[i].width), (0.3*faces[i].height));
        
        //extract mouth from face
        mouthROI = frame_gray( mouth_rect );
        
        threshold( mouthROI, mouthROI, threshold_value, 255, 1);
        
        //hold contour information
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Mat contourOutput = mouthROI.clone();
        Mat drawing = Mat::zeros( mouthROI.size(), CV_8UC3 );

        //search for contours
        findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

        //draw contours, find area of largest contour
        int max_area = 0;
        for( int i = 0; i< contours.size(); i++ )
        {
            if (contourArea(contours[i]) > max_area){
                max_area = contourArea(contours[i]);
            }
            Scalar color = Scalar( 0, 255, 0 );
            drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        }

        //display mouth ROI if mouth is open
        if (max_area > 2000){
            imshow("drawing", drawing);
            printf("mouth open, max area = %d\n", max_area);
        }

        //TODO:
        // add slider bar to adjust thresholds
        // draw circle around mouth in frame
        // how to send signl via USB?
    }

    //-- Show what we got
    imshow( window_name, frame);
}
