#include "c_api.h"
#include "VarFlow.h"
#include <highgui.h>
#include <opencv2/opencv.hpp>


void varflow(int width, int height, int max_level, int start_level, int n1, int n2,
                         float rho, float alpha, float sigma, void* U, void* V,
                         void* I1, void* I2) {
    VarFlow OpticalFlow(width, height, max_level, start_level, n1, n2, rho, alpha, sigma);
    IplImage* imgU = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_32F, 1);
    IplImage* imgV = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_32F, 1);
    IplImage* imgI1 = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_8U, 1);
    IplImage* imgI2 = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_8U, 1);
    cvSetData(imgU, static_cast<float*>(U), sizeof(float)*width);
    cvSetData(imgV, static_cast<float*>(V), sizeof(float)*width);
    cvSetData(imgI1, static_cast<unsigned char*>(I1), sizeof(unsigned char)*width);
    cvSetData(imgI2, static_cast<unsigned char*>(I2), sizeof(unsigned char)*width);
    OpticalFlow.CalcFlow(imgI1, imgI2, imgU, imgV, 0);
}