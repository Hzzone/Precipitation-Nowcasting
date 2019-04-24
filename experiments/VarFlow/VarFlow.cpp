//------------------------------------------------------------------
// Released under the BDS License
//
// Located at http://sourceforge.net/projects/varflow
//
//------------------------------------------------------------------

#include "VarFlow.h"
#include <iostream>

using namespace std;
/**
   Initializes all variables that don't need to get updated for each flow calculation.
   Note: Not much error checking is done, all inputs should be > 0

   @param[in] width_in   Width of images that will be used for calculation
   @param[in] height_in   Height of images that will be used for calculation
   @param[in] max_level_in   The maximum level that will be reached in the multigrid algorithm, higher maximum level = coarser level reached
   @param[in] start_level_in   The starting level used as the base in the multigrid algorithm, higher start level = coarser starting level
   @param[in] n1_in   Number of pre-smoothing steps in the multigrid cycle
   @param[in] n2_in   Number of post-smoothing steps in the multigrid cycle
   @param[in] rho_in   Gaussian smoothing parameter
   @param[in] alpha_in   Regularisation parameter in the energy functional
   @param[in] sigma_in   Gaussian smoothing parameter

*/
VarFlow::VarFlow(int width_in, int height_in, int max_level_in, int start_level_in, int n1_in, int n2_in,
                float rho_in, float alpha_in, float sigma_in){
					
	max_level = max_level_in;
    start_level = start_level_in;
    
    if(max_level < start_level)
    {
        max_level = start_level;
        cout<<"Warning: input max_level < start_level, correcting (new value = "<<max_level<<")"<<endl;
    }
	
	//Width and height of the largest image in the multigrid cycle, based on external input image dimensions
	//and the desired starting level
	int width = (int)floor(width_in/pow(2.0,(float)(start_level)));
    int height = (int)floor(height_in/pow(2.0,(float)(start_level)));
    
    // start_level too large, correct it
    if(width < 1 || height < 1)
    {
        if(width < 1)
        {
              start_level = (int)floor(log(width_in)/log(2));
              width = (int)floor(width_in/pow(2.0,(float)(start_level)));
              height = (int)floor(height_in/pow(2.0,(float)(start_level)));
        }
        
        if(height < 1)
        {
              start_level = (int)floor(log(height_in)/log(2));
              width = (int)floor(width_in/pow(2.0,(float)(start_level)));
              height = (int)floor(height_in/pow(2.0,(float)(start_level)));
        }
    
        // Correct max_level as well
        max_level = start_level;
        cout<<"Warning: start_level too large, correcting start_level and max_level (new value = "<<start_level<<")"<<endl;
        
    }
    
    int width_end = (int)floor(width_in/pow(2.0,(float)(max_level)));
    int height_end = (int)floor(height_in/pow(2.0,(float)(max_level)));
    
    // max_level too large, correct it
    if(width_end < 1 || height_end < 1)
    {
        if(width_end < 1)
        {
              max_level = (int)floor(log(width_in)/log(2));
              height_end = (int)floor(height_in/pow(2.0,(float)(max_level)));
        }
        
        if(height_end < 1)
        {
              max_level = (int)floor(log(height_in)/log(2));
        }
        
        cout<<"Warning: max_level too large, correcting (new value = "<<max_level<<")"<<endl;
        
    }
          
             
    n1 = n1_in;
    n2 = n2_in;
    
    rho = rho_in;
    alpha = alpha_in;
    sigma = sigma_in;
    
    // Spacial derivative masks
    mask_x[0] = 0.08333;
    mask_x[1] = -0.66666;
    mask_x[2] = 0;
    mask_x[3] = 0.66666;
    mask_x[4] = -0.08333;
    
    mask_y[0] = -0.08333;
    mask_y[1] = 0.66666;
    mask_y[2] = 0;
    mask_y[3] = -0.66666;
    mask_y[4] = 0.08333;
    
    fx_mask = cvMat(1, 5, CV_32F, mask_x);
    fy_mask = cvMat(5, 1, CV_32F, mask_y);
    
    //Resized input images will be stored in these variables
    imgAsmall = cvCreateImage(cvSize(width, height), 8, 1);
    imgBsmall = cvCreateImage(cvSize(width, height), 8, 1);
    
    //Float representations of resized input images
    imgAfloat = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    imgBfloat = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    
    //Spacial and temporal derivatives of input image A
    imgAfx = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    imgAfy = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    imgAft = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
    
    //Arrays to hold images of various sizes used in the multigrid cycle
    imgAfxfx_array = new IplImage*[max_level-start_level+1];  
    imgAfxfy_array = new IplImage*[max_level-start_level+1];  
    imgAfxft_array = new IplImage*[max_level-start_level+1];  
    imgAfyfy_array = new IplImage*[max_level-start_level+1];  
    imgAfyft_array = new IplImage*[max_level-start_level+1];  
    
    imgU_array = new IplImage*[max_level-start_level+1];  
    imgV_array = new IplImage*[max_level-start_level+1];  
    imgU_res_err_array = new IplImage*[max_level-start_level+1];  
    imgV_res_err_array = new IplImage*[max_level-start_level+1];  

    int i;
    
    //Allocate memory for image arrays
    for(i = 0; i < (max_level-start_level+1); i++){
        
        imgAfxfx_array[i] = cvCreateImage(cvSize((int)floor(width/pow(2.0,(float)(i))),(int)floor(height/pow(2.0,(float)(i)))), IPL_DEPTH_32F, 1);
        imgAfxfy_array[i] = cvCreateImage(cvSize((int)floor(width/pow(2.0,(float)(i))),(int)floor(height/pow(2.0,(float)(i)))), IPL_DEPTH_32F, 1);
        imgAfxft_array[i] = cvCreateImage(cvSize((int)floor(width/pow(2.0,(float)(i))),(int)floor(height/pow(2.0,(float)(i)))), IPL_DEPTH_32F, 1);
        imgAfyfy_array[i] = cvCreateImage(cvSize((int)floor(width/pow(2.0,(float)(i))),(int)floor(height/pow(2.0,(float)(i)))), IPL_DEPTH_32F, 1);
        imgAfyft_array[i] = cvCreateImage(cvSize((int)floor(width/pow(2.0,(float)(i))),(int)floor(height/pow(2.0,(float)(i)))), IPL_DEPTH_32F, 1);
    
        imgU_array[i] = cvCreateImage(cvSize((int)floor(width/pow(2.0,(float)(i))),(int)floor(height/pow(2.0,(float)(i)))), IPL_DEPTH_32F, 1);
        imgV_array[i] = cvCreateImage(cvSize((int)floor(width/pow(2.0,(float)(i))),(int)floor(height/pow(2.0,(float)(i)))), IPL_DEPTH_32F, 1);
        imgU_res_err_array[i] = cvCreateImage(cvSize((int)floor(width/pow(2.0,(float)(i))),(int)floor(height/pow(2.0,(float)(i)))), IPL_DEPTH_32F, 1);
        imgV_res_err_array[i] = cvCreateImage(cvSize((int)floor(width/pow(2.0,(float)(i))),(int)floor(height/pow(2.0,(float)(i)))), IPL_DEPTH_32F, 1);
    
        cvZero(imgU_array[i]);
        cvZero(imgV_array[i]);
        cvZero(imgU_res_err_array[i]);
        cvZero(imgV_res_err_array[i]);
       
    }
    
    initialized = 1;
    
}

/**
   Release all memory allocated for images
*/
VarFlow::~VarFlow(){
	
	cvReleaseImage(&imgAsmall);
    cvReleaseImage(&imgBsmall);
    
    cvReleaseImage(&imgAfloat);
    cvReleaseImage(&imgBfloat);
    
    cvReleaseImage(&imgAfx);
    cvReleaseImage(&imgAfy);
    cvReleaseImage(&imgAft);
    
    int i;
    
    for(i = 0; i < (max_level - start_level + 1); i++){
                     
            cvReleaseImage(&imgAfxfx_array[i]);
            cvReleaseImage(&imgAfxfy_array[i]);
            cvReleaseImage(&imgAfxft_array[i]);
            cvReleaseImage(&imgAfyfy_array[i]);
            cvReleaseImage(&imgAfyft_array[i]);
            
            cvReleaseImage(&imgU_array[i]);
            cvReleaseImage(&imgV_array[i]);
            cvReleaseImage(&imgU_res_err_array[i]);
            cvReleaseImage(&imgV_res_err_array[i]);
        
    }
    
    delete[] imgAfxfx_array;
    delete[] imgAfxfy_array;
    delete[] imgAfxft_array;
    delete[] imgAfyfy_array;
    delete[] imgAfyft_array;
    
    delete[] imgU_array;
    delete[] imgV_array;
    delete[] imgU_res_err_array;
    delete[] imgV_res_err_array;
    
    initialized = 0;
    
}

/**
   Implements the Gauss-Seidel method to upate the value of a flow field at a single pixel, equations 6 and 7 in Bruhn et al.
   
   Note that the function was written based on the calculation for the horizontal (u) flow field, thus all the variable names
   represent this fact. However, the same function is used for updating the vertical flow field, and the argument names passed
   in this case will not "match" the variable names in the function. For example, when updating the vertical field, the argument
   J11 will actually hold the value of J22, etc.

   @param[in] u   Flow field component currently being operated on (horizontal or vertical)
   @param[in] x   X coordinate of pixel being calculated
   @param[in] y   Y coordinate of pixel being calculated
   @param[in] h   Current pixel grid size
   @param[in] J11   The (1,1) component of the current structure tensor
   @param[in] J12   The (1,2) component of the current structure tensor
   @param[in] J13   The (1,3) component of the current structure tensor
   @param[in] vi   The value of the opposite flow field at the current pixel
   
   @return   The updated value of the flow field

*/
float VarFlow::gauss_seidel_step(IplImage* u, int x, int y, float h, float J11, float J12, float J13, float vi){
    
    int start_y, end_y, start_x, end_x;
    int N_num = 0;
                
    start_y = y - 1;
    end_y = y + 1;
    start_x = x - 1;
    end_x = x+1;         
                  
    float temp_u = 0;
            
    // Sum top neighbor    
    if(start_y > -1){              
        
        temp_u += *((float*)(u->imageData + start_y*u->widthStep) + x);
    
        N_num++;
     
    }
    
    // Sum bottom neighbor            
    if(end_y < u->height){   

        temp_u += *((float*)(u->imageData + end_y*u->widthStep) + x);
    
        N_num++;
              
    }
 
    // Sum left neighbor
    if(start_x > -1){              
                    
        temp_u += *((float*)(u->imageData + y*u->widthStep) + start_x);
    
        N_num++;
        
    }
    
    // Sum right neighbor
    if(end_x < u->width){              
                    
        temp_u += *((float*)(u->imageData + y*u->widthStep) + end_x);
    
        N_num++;
        
    }
    
    temp_u = temp_u - (h*h/alpha)*(J12*vi + J13);
    temp_u = temp_u / (N_num + (h*h/alpha)*J11);
                
                
    return temp_u;
    
}

/**
   Uses the Gauss-Seidel method to calculate the horizontal and vertical flow fields at a certain level in the multigrid
   process.

   @param[in] current_level   The current level of the multigrid algorithm (higher level = coarser)
   @param[in] h   Current pixel grid spacing
   @param[in] num_iter   Number of times to iterate at the current level
   @param[in] J13_array   Array of images representing the (1,3) component of the structure tensor
   @param[in] J23_array   Array of images representing the (2,3) component of the structure tensor

*/
void VarFlow::gauss_seidel_iteration(int current_level, float h, int num_iter, IplImage** J13_array, IplImage** J23_array){
                                
                                
	IplImage* imgU = imgU_array[current_level];    
	IplImage* imgV = imgV_array[current_level];  
	IplImage* imgAfxfx = imgAfxfx_array[current_level];
	IplImage* imgAfxfy = imgAfxfy_array[current_level];
	IplImage* imgAfxft = J13_array[current_level];
	IplImage* imgAfyfy = imgAfyfy_array[current_level];
	IplImage* imgAfyft = J23_array[current_level];
                                
    
	int i, k;
	int x;
	int y;
	float *u_ptr, *v_ptr, *fxfx_ptr, *fxfy_ptr, *fxft_ptr, *fyfy_ptr, *fyft_ptr;
	
	int max_i = imgU->height * imgU->width;
	
	u_ptr = (float*)(imgU->imageData);
	v_ptr = (float*)(imgV->imageData);
	        
	fxfx_ptr = (float*)(imgAfxfx->imageData);
	fxfy_ptr = (float*)(imgAfxfy->imageData);
	fxft_ptr = (float*)(imgAfxft->imageData);
	fyfy_ptr = (float*)(imgAfyfy->imageData);
	fyft_ptr = (float*)(imgAfyft->imageData);
	
	for(k = 0; k < num_iter; k++){
        
        x = 0;
        y = 0;
        
        for(i = 0; i < max_i; i++){
                  
               // Update flow fields
                u_ptr[i] = gauss_seidel_step(imgU, x, y, h, fxfx_ptr[i], fxfy_ptr[i], fxft_ptr[i], v_ptr[i]);              
                v_ptr[i] = gauss_seidel_step(imgV, x, y, h, fyfy_ptr[i], fxfy_ptr[i], fyft_ptr[i], u_ptr[i]);
               
                x++;
                if(x == imgU->width){
                  x = 0;
                  y++;
                }
        
              
        }  // End for loop, image traversal
    
    }  // End for loop, number of iterations
    
}


/**
   Calculates part of the value of a residual field at a single pixel
   
   Note that this function was written based on the calculation for the horizontal (u) residual field, so the remarks for
   gauss_seidel_step(...) apply here as well.
   
   Also note that this function doesn't calculate the "residual" field exactly, rather it calcualtes A^h * x_tilde ^ h from
   equation 10 of the paper by Bruhn et al. The final residual field is calculated by calculate_residual(...) which implements
   equation 10 fully.

   @param[in] u   Flow field component currently being operated on (horizontal or vertical)
   @param[in] x   X coordinate of pixel being calculated
   @param[in] y   Y coordinate of pixel being calculated
   @param[in] h   Current pixel grid size
   @param[in] J11   The (1,1) component of the current structure tensor
   @param[in] J12   The (1,2) component of the current structure tensor
   @param[in] vi   The value of the opposite flow field at the current pixel
   
   @return   Part of the residual field at the current coordinates

*/

float VarFlow::residual_part_step(IplImage* u, int x, int y, float h, float J11, float J12, float vi){
    
    int start_y, end_y, start_x, end_x;
        
    start_y = y - 1;
    end_y = y + 1;
    start_x = x - 1;
    end_x = x+1;
                
    float ih2 = 1 / (h*h);
                  
    float temp_res = 0;
    int N_num = 0;
    float curr_u = *((float*)(u->imageData + y*u->widthStep) + x);
                
    // Sum top neighbor
      
    if(start_y > -1){              
        
        temp_res += *((float*)(u->imageData + start_y*u->widthStep) + x);
        N_num++;
     
    }
                
    if(end_y < u->height){   // Sum bottom neighbor
                
        temp_res += *((float*)(u->imageData + end_y*u->widthStep) + x);
        N_num++;
              
    }
    
    // Sum left neighbor
      
    if(start_x > -1){              
        
        temp_res += *((float*)(u->imageData + y*u->widthStep) + start_x);
        N_num++;
        
    }
    
    // Sum right neighbor
    
    if(end_x < u->width){              
        
        temp_res += *((float*)(u->imageData + y*u->widthStep) + end_x);
        N_num++;
       
    }
    
    temp_res = N_num*curr_u - temp_res;
    
    temp_res *= ih2;
    
    temp_res -= (1/alpha)*(J11*curr_u + J12*vi);
    
    return temp_res;
}

/**
   Calculates the full residual of the current flow field based on equation 10 in Bruhn et al.

   @param[in] current_level   The current level of the multigrid algorithm (higher level = coarser)
   @param[in] h   Current pixel grid spacing
   @param[in] J13_array   Array of images representing the (1,3) component of the structure tensor
   @param[in] J23_array   Array of images representing the (2,3) component of the structure tensor

*/

void VarFlow::calculate_residual(int current_level, float h, IplImage** J13_array, IplImage** J23_array){
								
                            
	IplImage* imgU = imgU_array[current_level];
	IplImage* imgV = imgV_array[current_level];
	IplImage* imgAfxfx = imgAfxfx_array[current_level];
	IplImage* imgAfxfy = imgAfxfy_array[current_level];
	IplImage* imgAfxft = J13_array[current_level];
	IplImage* imgAfyfy = imgAfyfy_array[current_level];
	IplImage* imgAfyft = J23_array[current_level];
	IplImage* imgU_res_err = imgU_res_err_array[current_level];
	IplImage* imgV_res_err = imgV_res_err_array[current_level];
                                
    int i;
    float *u_ptr, *v_ptr, *fxfx_ptr, *fxfy_ptr, *fyfy_ptr, *u_res_err_ptr, *v_res_err_ptr;
    
    int max_i = imgU->height * imgU->width;
    int x, y;
    
    u_res_err_ptr = (float*)(imgU_res_err->imageData);
    v_res_err_ptr = (float*)(imgV_res_err->imageData);
        
    u_ptr = (float*)(imgU->imageData);
    v_ptr = (float*)(imgV->imageData);
            
    fxfx_ptr = (float*)(imgAfxfx->imageData);
    fxfy_ptr = (float*)(imgAfxfy->imageData);
    fyfy_ptr = (float*)(imgAfyfy->imageData);
    
    x = 0;
    y = 0;
    
    for(i = 0; i < max_i; i++){
            
            // Get A^h * x_tilde^h (equation 10)
            u_res_err_ptr[i] = residual_part_step(imgU, x, y, h, fxfx_ptr[i], fxfy_ptr[i], v_ptr[i] );
            v_res_err_ptr[i] = residual_part_step(imgV, x, y, h, fyfy_ptr[i], fxfy_ptr[i], u_ptr[i] );
            
            x++;
            if(x == imgU->width){
                  x = 0;
                  y++;
            }
        
    }
    
    // Get full residual
    cvAddWeighted( imgAfxft, (1/alpha), imgU_res_err, -1, 0, imgU_res_err );
    cvAddWeighted( imgAfyft, (1/alpha), imgV_res_err, -1, 0, imgV_res_err );
}


/**
   This recursive function implements two V cycles of the Gauss-Seidel algorithm to calculate the flow field at a given level.
   One V cycle calculates the flow field, then the residual, applying a restriction operator to the residual, which then
   becomes the input to the next stage of the recursive algorithm. Once the maximum level is reached, the function propagates
   back up to the start, comleting one V cycle. The process is then repeated for the second V cycle.
   
   @param[in] current_level   The current level of the multigrid algorithm (higher level = coarser)
   @param[in] max_level   The maximum level the algorithm will reach
   @param[in] first_level   The starting level
   @param[in] h   Current pixel grid spacing
   @param[in] J13_array   Array of images representing the (1,3) component of the structure tensor
   @param[in] J23_array   Array of images representing the (2,3) component of the structure tensor

*/

void VarFlow::gauss_seidel_recursive(int current_level, int max_level, int first_level, float h, 
										IplImage** J13_array, IplImage** J23_array){
                                
    if(current_level == max_level){
         
        // Iterate normally n1 times and that's it
        gauss_seidel_iteration(current_level, h,  n1, J13_array, J23_array);
                     
    }
    
    else{
        
        //---------------------------- Start 1st V cycle -------------------------------------
     
        // Iterate n1 times
        gauss_seidel_iteration(current_level, h, n1, J13_array, J23_array); 
        
        // Calculate residual
        calculate_residual(current_level, h, J13_array, J23_array);
                               
        // Apply restriction operator to residual
        cvResize(imgU_res_err_array[current_level], imgU_res_err_array[current_level+1], CV_INTER_LINEAR);
        cvResize(imgV_res_err_array[current_level], imgV_res_err_array[current_level+1], CV_INTER_LINEAR);
        
        // Initialize new u and v images to zero
        cvZero(imgU_array[current_level+1]);
        cvZero(imgV_array[current_level+1]);
        
        // Pass residual down recursively (Important: switch J13 and J23 with imgU_res_err and imgV_res_err, increase h!!)  
        gauss_seidel_recursive(current_level+1, max_level, first_level, 2*h, imgU_res_err_array, imgV_res_err_array);
                 
        // Prolong solution to get error at current level                            
        cvResize(imgU_array[current_level+1], imgU_res_err_array[current_level], CV_INTER_LINEAR);
        cvResize(imgV_array[current_level+1], imgV_res_err_array[current_level], CV_INTER_LINEAR);
        
        // Correct original solution with error
        cvAdd(imgU_array[current_level], imgU_res_err_array[current_level], imgU_array[current_level], NULL);
        cvAdd(imgV_array[current_level], imgV_res_err_array[current_level], imgV_array[current_level], NULL);
        
        // Iterate n1+n2 times to smooth new solution
        gauss_seidel_iteration(current_level, h, n1+n2, J13_array, J23_array); 
        
                               
       //---------------------------- End 1st V cycle, Start 2nd V cycle -------------------------------------                        
                          
        // Calculate residual again
        calculate_residual(current_level,h, J13_array, J23_array);
                               
        // Apply restriction operator to residual
        cvResize(imgU_res_err_array[current_level], imgU_res_err_array[current_level+1], CV_INTER_LINEAR);
        cvResize(imgV_res_err_array[current_level], imgV_res_err_array[current_level+1], CV_INTER_LINEAR);
        
        // Initialize new u and v images to zero
        cvZero(imgU_array[current_level+1]);
        cvZero(imgV_array[current_level+1]);
        
        
        // Pass residual down recursively (Important: switch J13 and J23 with imgU_res_err and imgV_res_err, increase h!!)      
        gauss_seidel_recursive(current_level+1, max_level, first_level, 2*h, imgU_res_err_array, imgV_res_err_array);
               
        // Prolong solution to get error at current level                            
        cvResize(imgU_array[current_level+1], imgU_res_err_array[current_level], CV_INTER_LINEAR);
        cvResize(imgV_array[current_level+1], imgV_res_err_array[current_level], CV_INTER_LINEAR);
        
        // Correct original solution with error
        cvAdd(imgU_array[current_level], imgU_res_err_array[current_level], imgU_array[current_level], NULL);
        cvAdd(imgV_array[current_level], imgV_res_err_array[current_level], imgV_array[current_level], NULL);
        
        // Iterate n2 times to smooth new solution
        gauss_seidel_iteration(current_level, h, n2, J13_array, J23_array); 
        
                               
        //---------------------------- End 2nd V cycle -------------------------------------
              
    }
                                
}



/**
   Calculates the optical flow between two images.

   @param[in] imgA   First input image
   @param[in] imgB   Second input image, the flow is calculated from imgA to imgB
   @param[out] imgU   Horizontal flow field
   @param[out] imgV   Vertical flow field
   @param[in] saved_data   Flag indicates previous imgB is now imgA (ie subsequent frames), save some time in calculation
   
   @return   Flag to indicate succesful completion

*/
int VarFlow::CalcFlow(IplImage* imgA, IplImage* imgB, IplImage* imgU, IplImage* imgV, bool saved_data = false){
    
    if(!initialized)
      return 0;
      
    IplImage* swap_img;
      
    //Don't recalculate imgAfloat, just swap with imgBfloat
    if(saved_data){
        
       CV_SWAP(imgAfloat, imgBfloat, swap_img);
       
       cvResize(imgB, imgBsmall, CV_INTER_LINEAR);
       cvConvert(imgBsmall, imgBfloat);  // Calculate new imgBfloat
       cvSmooth(imgBfloat, imgBfloat, CV_GAUSSIAN, 0, 0, sigma, 0 );
       
    }
    
    //Calculate imgAfloat as well as imgBfloat
    else{
    
        cvResize(imgA, imgAsmall, CV_INTER_LINEAR);
        cvResize(imgB, imgBsmall, CV_INTER_LINEAR);
    
        cvConvert(imgAsmall, imgAfloat);
        cvConvert(imgBsmall, imgBfloat);
    
        cvSmooth(imgAfloat, imgAfloat, CV_GAUSSIAN, 0, 0, sigma, 0 );
        cvSmooth(imgBfloat, imgBfloat, CV_GAUSSIAN, 0, 0, sigma, 0 );
        
    }
    
    cvFilter2D(imgAfloat, imgAfx, &fx_mask, cvPoint(-1,-1));  // X spacial derivative
    cvFilter2D(imgAfloat, imgAfy, &fy_mask, cvPoint(-1,-1));  // Y spacial derivative
    
    cvSub(imgBfloat, imgAfloat, imgAft, NULL);  // Temporal derivative
    
    cvMul(imgAfx,imgAfx,imgAfxfx_array[0], 1);
    cvMul(imgAfx,imgAfy,imgAfxfy_array[0], 1);
    cvMul(imgAfx,imgAft,imgAfxft_array[0], 1);
    cvMul(imgAfy,imgAfy,imgAfyfy_array[0], 1);
    cvMul(imgAfy,imgAft,imgAfyft_array[0], 1);
    
    cvSmooth(imgAfxfx_array[0], imgAfxfx_array[0], CV_GAUSSIAN, 0, 0, rho, 0 );
    cvSmooth(imgAfxfy_array[0], imgAfxfy_array[0], CV_GAUSSIAN, 0, 0, rho, 0 );
    cvSmooth(imgAfxft_array[0], imgAfxft_array[0], CV_GAUSSIAN, 0, 0, rho, 0 );
    cvSmooth(imgAfyfy_array[0], imgAfyfy_array[0], CV_GAUSSIAN, 0, 0, rho, 0 );
    cvSmooth(imgAfyft_array[0], imgAfyft_array[0], CV_GAUSSIAN, 0, 0, rho, 0 );
    
    int i;
    
    //Fill all the levels of the multigrid algorithm with resized images
    for(i = 1; i < (max_level - start_level + 1); i++){
        
        cvResize(imgAfxfx_array[i-1], imgAfxfx_array[i], CV_INTER_LINEAR);
        cvResize(imgAfxfy_array[i-1], imgAfxfy_array[i], CV_INTER_LINEAR);
        cvResize(imgAfxft_array[i-1], imgAfxft_array[i], CV_INTER_LINEAR);
        cvResize(imgAfyfy_array[i-1], imgAfyfy_array[i], CV_INTER_LINEAR);
        cvResize(imgAfyft_array[i-1], imgAfyft_array[i], CV_INTER_LINEAR);
        
    }
    
    int k = (max_level - start_level);

    while(1){
    
        gauss_seidel_recursive(k, (max_level - start_level), k, pow(2.0,(float)(k)), imgAfxft_array, imgAfyft_array);
    
        
        if(k > 0){
            
            // Transfer velocity from coarse to fine                           
            cvResize(imgU_array[k], imgU_array[k-1], CV_INTER_LINEAR);
            cvResize(imgV_array[k], imgV_array[k-1], CV_INTER_LINEAR);
            
            k--;
            
        }
        else
          break;
    
        
    }
    
    // Transfer to output image, resize if necessary
    cvResize(imgU_array[0], imgU, CV_INTER_LINEAR);
    cvResize(imgV_array[0], imgV, CV_INTER_LINEAR);
    
    // If started algorithm with smaller image than original, scale the flow field
    if(start_level > 0){
	
		cvScale(imgU, imgU, pow(2.0, start_level));
		cvScale(imgV, imgV, pow(2.0, start_level));
		
	}
    
    return 1;
}
