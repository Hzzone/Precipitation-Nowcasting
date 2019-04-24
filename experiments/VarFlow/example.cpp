//------------------------------------------------------------------
// Released under the BDS License
//
// Located at http://sourceforge.net/projects/varflow
//
//------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------
//
// Sample program demonstrating usage of the VarFlow class to calculate optical flow
// Two images from the classic Yosemite flyby sequence are used to generate a dense optical flow field.
// The original image is displayed, along with vector and color field representations of the underlying flow field.
//
// Author: Adam Harmat
// Date: April 2009
//
//----------------------------------------------------------------------------------------------------------


#include <highgui.h>
#include <iostream>
#include "VarFlow.h" 
//#include "ProfTimer.h"

using namespace std;

#define M_PI       3.14159265358979323846   // pi

// Calculates RGB values from HSV color space
void hsv2rgb(float h, float s, float v, uchar &r, uchar &g, uchar &b){
     
     if(h > 360)
     {
        h = h - 360;
     }
     
     float c = v*s;   // chroma
     float hp = h / 60;
     
     float hpmod2 = hp - (float)((int)(hp/2))*2;
     
     float x = c*(1 - fabs(hpmod2 - 1));
     float m = v - c;
     
     float r1, g1, b1;
     
     if(0 <= hp && hp < 1){
          r1 = c;
          g1 = x;
          b1 = 0;
     }
     else if(1 <= hp && hp < 2){
          r1 = x;
          g1 = c;
          b1 = 0;
     }
     else if(2 <= hp && hp < 3){
          r1 = 0;
          g1 = c;
          b1 = x;
     }
     else if(3 <= hp && hp < 4){
          r1 = 0;
          g1 = x;
          b1 = c;
     }
     else if(4 <= hp && hp < 5){
          r1 = x;
          g1 = 0;
          b1 = c;
     }
     else
     {
          r1 = c;
          g1 = 0;
          b1 = x;
     }
     
     r = (uchar)(255*(r1 + m));
     g = (uchar)(255*(g1 + m));
     b = (uchar)(255*(b1 + m));
     
}

// Draw a vector field based on horizontal and vertical flow fields
void drawMotionField(IplImage* imgU, IplImage* imgV, IplImage* imgMotion, int xSpace, int ySpace, float cutoff, int multiplier, CvScalar color)
{
     int x, y;
    
     CvPoint p0 = cvPoint(0,0);
     CvPoint p1 = cvPoint(0,0);
     
     float deltaX, deltaY, angle, hyp;
     
     for(y = ySpace; y < imgU->height; y+= ySpace ) {
        for(x = xSpace; x < imgU->width; x+= xSpace ){
         
            p0.x = x;
            p0.y = y;
            
            deltaX = *((float*)(imgU->imageData + y*imgU->widthStep)+x);
            deltaY = -(*((float*)(imgV->imageData + y*imgV->widthStep)+x));
            
            angle = atan2(deltaY, deltaX);
            hyp = sqrt(deltaX*deltaX + deltaY*deltaY);
   
            if(hyp > cutoff){
                   
                p1.x = p0.x + cvRound(multiplier*hyp*cos(angle));
                p1.y = p0.y + cvRound(multiplier*hyp*sin(angle));
                   
                cvLine( imgMotion, p0, p1, color,1, CV_AA, 0);
                
                p0.x = p1.x + cvRound(3*cos(angle-M_PI + M_PI/4));
                p0.y = p1.y + cvRound(3*sin(angle-M_PI + M_PI/4));
                cvLine( imgMotion, p0, p1, color,1, CV_AA, 0);
                
                p0.x = p1.x + cvRound(3*cos(angle-M_PI - M_PI/4));
                p0.y = p1.y + cvRound(3*sin(angle-M_PI - M_PI/4));
                cvLine( imgMotion, p0, p1, color,1, CV_AA, 0);
            }
      
        }
    }
    
}

// Draws the circular legend for the color field, indicating direction and magnitude
void drawLegendHSV(IplImage* imgColor, int radius, int cx, int cy)
{
     int width = radius*2 + 1;
     int height = width;
     
     IplImage* imgLegend = cvCreateImage( cvSize(width, height), 8, 3 );
     IplImage* imgMask = cvCreateImage( cvSize(width, height), 8, 1 );
     IplImage* sub_img = cvCreateImageHeader(cvSize(width, height),8,3);
     
     uchar* legend_ptr;
     float angle, h, s, v, legend_max_s;
     uchar r,g,b;
     int deltaX, deltaY;
     
     legend_max_s = radius*sqrt(2);
     
     for(int y=0; y < imgLegend->height; y++)
     {
         legend_ptr = (uchar*)(imgLegend->imageData + y*imgLegend->widthStep);
         
         for(int x=0; x < imgLegend->width; x++)
         {
              deltaX = x-radius;
              deltaY = -(y-radius);
              angle = atan2(deltaY,deltaX);
              
              if(angle < 0)
                 angle += 2*M_PI;
                   
              h = angle * 180 / M_PI;
              s = sqrt(deltaX*deltaX + deltaY*deltaY) / legend_max_s;
              v = 0.9;
              
              hsv2rgb(h, s, v, r, g, b);
               
              legend_ptr[3*x] = b;
              legend_ptr[3*x+1] = g;
              legend_ptr[3*x+2] = r;
                   
         }
     }
     
     cvZero(imgMask);
     cvCircle( imgMask, cvPoint(radius,radius) , radius, CV_RGB(255,255,255), -1,8,0 );
     
     sub_img->origin = imgColor->origin;
     sub_img->widthStep = imgColor->widthStep;
     sub_img->imageData = imgColor->imageData + (cy-radius) * imgColor->widthStep + (cx-radius) * imgColor->nChannels;
	        
     cvCopy(imgLegend, sub_img, imgMask);
     
     cvCircle( imgColor, cvPoint(cx,cy) , radius, CV_RGB(0,0,0), 1,CV_AA,0 );
     
     cvReleaseImage(&imgLegend);
     cvReleaseImage(&imgMask);
     cvReleaseImageHeader(&sub_img);
     
}


// Draws a color field representation of the flow field
void drawColorField(IplImage* imgU, IplImage* imgV, IplImage* imgColor)  
{
    IplImage* imgColorHSV = cvCreateImage( cvSize(imgColor->width, imgColor->height), IPL_DEPTH_32F, 3 );
    cvZero(imgColorHSV);
    
    float max_s = 0;
    float *hsv_ptr, *u_ptr, *v_ptr;
    uchar *color_ptr;
    float angle;
    float h,s,v;
    uchar r,g,b;
    float deltaX, deltaY;
    
    int x, y;
   
    // Generate hsv image
    for(y = 0; y < imgColorHSV->height; y++ ) {
		
		hsv_ptr = (float*)(imgColorHSV->imageData + y*imgColorHSV->widthStep);
		u_ptr = (float*)(imgU->imageData + y*imgU->widthStep);
		v_ptr = (float*)(imgV->imageData + y*imgV->widthStep);
		
		for(x = 0; x < imgColorHSV->width; x++){
              
              deltaX = u_ptr[x];
              deltaY = v_ptr[x];
                 
              angle = atan2(deltaY,deltaX);
              
              if(angle < 0)
                 angle += 2*M_PI;
                   
              hsv_ptr[3*x] = angle * 180 / M_PI;
              hsv_ptr[3*x+1] = sqrt(deltaX*deltaX + deltaY*deltaY);
              hsv_ptr[3*x+2] = 0.9;	
              
              if(hsv_ptr[3*x+1] > max_s)
                 max_s = hsv_ptr[3*x+1];
        
        }
    }
  
     // Generate color image
    for(y = 0; y < imgColor->height; y++ ) {
		
		hsv_ptr = (float*)(imgColorHSV->imageData + y*imgColorHSV->widthStep);
		color_ptr = (uchar*)(imgColor->imageData + y*imgColor->widthStep);
		
		for(x = 0; x < imgColor->width; x++){
              
              h = hsv_ptr[3*x];
              s = hsv_ptr[3*x+1] / max_s;
              v = hsv_ptr[3*x+2];
              
              hsv2rgb(h, s, v, r, g, b);
               
              color_ptr[3*x] = b;
              color_ptr[3*x+1] = g;
              color_ptr[3*x+2] = r;
              
       }
    }
    
    drawLegendHSV(imgColor, 25, 28, 28);
    
    cvReleaseImage(&imgColorHSV);
     
}
    

int main(int argc, char *argv[])
{
    // Load sample images
    IplImage* imgA = cvLoadImage( "Data/yos_img_08.jpg", 0 );
    IplImage* imgB = cvLoadImage( "Data/yos_img_09.jpg", 0 );
    
    int width = imgA->width;
    int height = imgA->height;

    IplImage* imgU = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    IplImage* imgV = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    IplImage* imgColor = cvCreateImage( cvSize(width, height), 8, 3 );
    IplImage* imgMotion = cvCreateImage(cvSize(width, height), 8, 3);
      
    IplImage* swap_img;
    
    //We will start at level 0 (full size image) and go down to level 4 (coarse image 16 times smaller than original)
    //Experiment with these values to see how they affect the flow field as well as calculation time
    int max_level = 4;
    int start_level = 0;

    //Two pre and post smoothing steps, should be greater than zero
    int n1 = 2;
    int n2 = 2;

    //Smoothing and regularization parameters, experiment but keep them above zero
    float rho = 2.8;
    float alpha = 1400;
    float sigma = 1.5;
    
    char c;
    
    // Set up VarFlow class
    VarFlow OpticalFlow(width, height, max_level, start_level, n1, n2, rho, alpha, sigma);

    //ProfTimer t;
 
    cvZero(imgU);
    cvZero(imgV);
    cvZero(imgMotion);
    cvZero(imgColor);
    
    // Start timing    
    //t.Start();
         
    // Calculate the flow
    OpticalFlow.CalcFlow(imgA, imgB, imgU, imgV, 0);
    
    // Stop timing
    //t.Stop();
    //double dur = t.GetDurationInSecs();
    //cout<<"Executing CalcFlow took "<<dur<<" seconds"<<endl;
    
    cvNamedWindow("Motion",CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Original",CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Color",CV_WINDOW_AUTOSIZE);
    
    // Draw motion field with grid spacing of 10, minimium displacement 1 pixel, arrow length multiplier of 5         
    drawMotionField(imgU, imgV, imgMotion, 10, 10, 1, 5, CV_RGB(255,0,0));
    drawColorField(imgU, imgV, imgColor);                 
       
    cvShowImage("Color", imgColor);
    cvShowImage("Motion", imgMotion);
    cvShowImage("Original", imgA);
    
    cout<<"Press ESC to quit"<<endl;
   
   while(1){ 
        
        c = cvWaitKey(500);
        
        CV_SWAP(imgA, imgB, swap_img);
        cvShowImage("Original", imgA);
        
        if( c == 27 ) break;
    }
    
    
    cvDestroyWindow("Color");
    cvDestroyWindow("Motion");
    cvDestroyWindow("Original");
    
    cvReleaseImage(&imgA);
    cvReleaseImage(&imgB);
    cvReleaseImage(&imgU);
    cvReleaseImage(&imgV);
    cvReleaseImage(&imgColor);
    cvReleaseImage(&imgMotion);
    
 
    system("PAUSE");
    
    return 0;
}
