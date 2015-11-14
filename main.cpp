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

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */

String face_cascade_name = "/Users/mchapman/Downloads/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
String window_name = "Capture";

/** @function main */
int main( void )
{
    VideoCapture capture;
    Mat frame;
    
    //-- 1. Load the face cascade
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    
    //-- 2. Read the video stream
    capture.open( 0 );
    printf("opened webcam\n");

    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
 
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

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    
    //convert to grayscale, equalize distribution of pixels
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        //draw ellipse around face
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        
        Mat faceROI = frame_gray( faces[i] );
        
        //ROI where mouth will be
        Rect mouth_rect = Rect(faces[i].x + (0.25*faces[i].width), faces[i].y + (0.6*faces[i].height), (0.5*faces[i].width), (0.35*faces[i].height));
        
        //extract mouth from face
        Mat mouthROI = frame_gray( mouth_rect );
        
        threshold( mouthROI, mouthROI, 95, 255,0);
        
        //display mouth ROI
        imshow("mouth", mouthROI);
    }

    //-- Show what we got
    imshow( window_name, frame);
}
