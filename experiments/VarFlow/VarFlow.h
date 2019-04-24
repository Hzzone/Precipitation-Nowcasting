//------------------------------------------------------------------
// Released under the BDS License
//
// Located at http://sourceforge.net/projects/varflow
//
//------------------------------------------------------------------

#ifndef VARFLOW_H
#define VARFLOW_H

#include <cv.h>
#include <cxcore.h>

/**
 * @brief Calculates dense optical flow using a variational method.
 *
 * The VarFlow class implements the method described in "Real-Time Optic Flow Computation with Variational Methods" by 
 * Bruhn et al. (Lecture Notes in Computer Science, Volume 2756/2003, pp 222-229). It uses a recursive multigrid algorithm to 
 * minimize the energy functional, leading to increased performance. This implementation uses an uncoupled Gauss-Seidel algorithm
 * and the temporal derivative of an image is calculated using a simple difference instead of a two point stencil as described 
 * in the original paper.
 *
 * Date: April 2009
 *
 * @author Adam Harmat
 */


class VarFlow{

    public:
    
        VarFlow(int width_in, int height_in, int max_level_in, int start_level_in, int n1_in, int n2_in,
        float rho_in, float alpha_in, float sigma_in);
        ~VarFlow();
        int CalcFlow(IplImage* imgA, IplImage* imgB, IplImage* imgU, IplImage* imgV, bool saved_data);
    
    private:
    
        void gauss_seidel_recursive(int current_level, int max_level, int first_level, float h, IplImage** J13_array, IplImage** J23_array);
        void gauss_seidel_iteration(int current_level, float h, int num_iter, IplImage** J13_array, IplImage** J23_array);
        float gauss_seidel_step(IplImage* u, int x, int y, float h, float J11, float J12, float J13, float vi);	
        float residual_part_step(IplImage* u, int x, int y, float h, float J11, float J12, float vi);
        void calculate_residual(int current_level, float h, IplImage** J13_array, IplImage** J23_array);
                    
        
        float mask_x[5];
        float mask_y[5];
        CvMat fx_mask;
        CvMat fy_mask;
        
        IplImage* imgAsmall;
        IplImage* imgBsmall;
        
        IplImage* imgAfloat;
        IplImage* imgBfloat;
        
        IplImage* imgAfx;
        IplImage* imgAfy;
        IplImage* imgAft;
        
        IplImage** imgAfxfx_array;  
        IplImage** imgAfxfy_array; 
        IplImage** imgAfxft_array; 
        IplImage** imgAfyfy_array; 
        IplImage** imgAfyft_array; 
        
        IplImage** imgU_array;
        IplImage** imgV_array;
        IplImage** imgU_res_err_array;
        IplImage** imgV_res_err_array;
        
        int initialized;
        
        int max_level;
        int start_level;
        
        int n1;
        int n2;
        
        float rho;
        float alpha;
        float sigma;
};


#endif
