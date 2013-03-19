/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: superresolution
* file:    flowlib_gpu_sor.cu
*
*
* implement all functions with ### implement me ### in the function body
\****************************************************************************/

/*
 * flowlib_gpu_sor.cu
 *
 *  Created on: Mar 14, 2012
 *      Author: steinbrf
 */

//#include <flowlib_gpu_sor.hpp>
#include "flowlib.hpp"
#include <auxiliary/cuda_basic.cuh>
#include <linearoperations/linearoperations.cuh>
#include <auxiliary/debug.hpp>

cudaChannelFormatDesc flow_sor_float_tex = cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_flow_sor_I1;
texture<float, 2, cudaReadModeElementType> tex_flow_sor_I2;
bool textures_flow_sor_initialized = false;

#define IMAGE_FILTER_METHOD cudaFilterModeLinear
#define SF_TEXTURE_OFFSET 0.5f

#define SF_BW 16
#define SF_BH 16


FlowLibGpuSOR::FlowLibGpuSOR(int par_nx, int par_ny):
FlowLib(par_nx,par_ny),FlowLibGpu(par_nx,par_ny),FlowLibSOR(par_nx,par_ny)
{

	cuda_malloc2D((void**)&_penDat,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_penReg,_nx,_ny,1,sizeof(float),&_pitchf1);

	cuda_malloc2D((void**)&_b1,_nx,_ny,1,sizeof(float),&_pitchf1);
	cuda_malloc2D((void**)&_b2,_nx,_ny,1,sizeof(float),&_pitchf1);

}

FlowLibGpuSOR::~FlowLibGpuSOR()
{
	if(_penDat) cutilSafeCall(cudaFree(_penDat));
	if(_penReg) cutilSafeCall(cudaFree(_penReg));
	if(_b1)     cutilSafeCall(cudaFree(_b1));
	if(_b2)     cutilSafeCall(cudaFree(_b2));
}

void bind_textures(const float *I1_g, const float *I2_g, int nx, int ny, int pitchf1)
{
	tex_flow_sor_I1.addressMode[0] = cudaAddressModeClamp;
	tex_flow_sor_I1.addressMode[1] = cudaAddressModeClamp;
	tex_flow_sor_I1.filterMode = IMAGE_FILTER_METHOD ;
	tex_flow_sor_I1.normalized = false;

	tex_flow_sor_I2.addressMode[0] = cudaAddressModeClamp;
	tex_flow_sor_I2.addressMode[1] = cudaAddressModeClamp;
	tex_flow_sor_I2.filterMode = IMAGE_FILTER_METHOD;
	tex_flow_sor_I2.normalized = false;

	cutilSafeCall( cudaBindTexture2D(0, &tex_flow_sor_I1, I1_g,
		&flow_sor_float_tex, nx, ny, pitchf1*sizeof(float)) );
	cutilSafeCall( cudaBindTexture2D(0, &tex_flow_sor_I2, I2_g,
		&flow_sor_float_tex, nx, ny, pitchf1*sizeof(float)) );
}

void unbind_textures_flow_sor()
{
  cutilSafeCall (cudaUnbindTexture(tex_flow_sor_I1));
  cutilSafeCall (cudaUnbindTexture(tex_flow_sor_I2));
}

void update_textures_flow_sor(const float *I2_resampled_warped_g, int nx_fine, int ny_fine, int pitchf1)
{
	cutilSafeCall (cudaUnbindTexture(tex_flow_sor_I2));
	cutilSafeCall( cudaBindTexture2D(0, &tex_flow_sor_I2, I2_resampled_warped_g,
		&flow_sor_float_tex, nx_fine, ny_fine, pitchf1*sizeof(float)) );
}


/**
 * @brief Adds one flow field onto another
 * @param du_g Horizontal increment
 * @param dv_g Vertical increment
 * @param u_g Horizontal accumulation
 * @param v_g Vertical accumulation
 * @param nx Image width
 * @param ny Image height
 * @param pitchf1 Image pitch for single float images
 */
__global__ void add_flow_fields
(
	const float *du_g,
	const float *dv_g,
	float *u_g,
	float *v_g,
	int    nx,
	int    ny,
	int    pitchf1
)
{
	// ### Implement Me###
}


/**
 * @brief Kernel to compute the penalty values for several
 * lagged-diffusivity iterations taking into account pixel sizes for warping.
 * Image derivatives are read from texture, flow derivatives from shared memory
 * @param u_g Pointer to global device memory for the horizontal
 * flow component of the accumulation flow field
 * @param v_g Pointer to global device memory for the vertical
 * flow component of the accumulation flow field
 * @param du_g Pointer to global device memory for the horizontal
 * flow component of the increment flow field
 * @param dv_g Pointer to global device memory for the vertical
 * flow component of the increment flow field
 * @param penaltyd_g Pointer to global device memory for data term penalty
 * @param penaltyr_g Pointer to global device memory for regularity term
 * penalty
 * @param nx Image width
 * @param ny Image height
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param data_epsilon Smoothing parameter for the TV Penalization of the data
 * term
 * @param diff_epsilon Smoothing parameter for the TV Penalization of the
 * regularity term
 * @param pitchf1 Image pitch for single float images
 */
__global__ void sorflow_update_robustifications_warp_tex_shared
(
	const float *u_g,
	const float *v_g,
	const float *du_g,
	const float *dv_g,
	float *penaltyd_g,
	float *penaltyr_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  data_epsilon,
	float  diff_epsilon,
	int    pitchf1
)
{
	  const int x = blockIdx.x * blockDim.x + threadIdx.x;
	  const int y = blockIdx.y * blockDim.y + threadIdx.y;
	  const int tx = threadIdx.x + 1;
	  const int ty = threadIdx.y + 1;
	  const int idx = y*pitchf1 + x;

	  __shared__ float s_u1[SF_BW+2][SF_BH+2];
	  __shared__ float s_u2[SF_BW+2][SF_BH+2];
	  __shared__ float s_u1lvl[SF_BW+2][SF_BH+2];
	  __shared__ float s_u2lvl[SF_BW+2][SF_BH+2];
	  
	  // load data into shared memory
	  if (x < nx && y < ny) {
	    s_u1[tx][ty] = u_g[idx];
	    s_u2[tx][ty] = v_g[idx];
	    s_u1lvl[tx][ty] = du_g[idx];
	    s_u2lvl[tx][ty] = dv_g[idx];
	    
	    if (x == 0) {
		    s_u1[0][ty] = s_u1[tx][ty];
		    s_u2[0][ty] = s_u2[tx][ty];
		    s_u1lvl[0][ty] = s_u1lvl[tx][ty];
		    s_u2lvl[0][ty] = s_u2lvl[tx][ty];
	    }
	    else if (threadIdx.x == 0) {
		    s_u1[0][ty] = u_g[idx-1];
		    s_u2[0][ty] = v_g[idx-1];
		    s_u1lvl[0][ty] = du_g[idx-1];
		    s_u2lvl[0][ty] = dv_g[idx-1];
	    }
	      
	    if (x == nx-1) {
		    s_u1[tx+1][ty] = s_u1[tx][ty];
		    s_u2[tx+1][ty] = s_u2[tx][ty];
		    s_u1lvl[tx+1][ty] = s_u1lvl[tx][ty];
		    s_u2lvl[tx+1][ty] = s_u2lvl[tx][ty];
	    }
	    else if (threadIdx.x == blockDim.x-1) {
		    s_u1[tx+1][ty] = u_g[idx+1];
		    s_u2[tx+1][ty] = v_g[idx+1];
		    s_u1lvl[tx+1][ty] = du_g[idx+1];
		    s_u2lvl[tx+1][ty] = dv_g[idx+1];
	    }

	    if (y == 0) {
		    s_u1[tx][0] = s_u1[tx][ty];
		    s_u2[tx][0] = s_u2[tx][ty];
		    s_u1lvl[tx][0] = s_u1lvl[tx][ty];
		    s_u2lvl[tx][0] = s_u2lvl[tx][ty];
	    }
	    else if (threadIdx.y == 0) {
		    s_u1[tx][0] = u_g[idx-pitchf1];
		    s_u2[tx][0] = v_g[idx-pitchf1];
		    s_u1lvl[tx][0] = du_g[idx-pitchf1];
		    s_u2lvl[tx][0] = dv_g[idx-pitchf1];
	    }
	      
	    if (y == ny-1) {
		    s_u1[tx][ty+1] = s_u1[tx][ty];
		    s_u2[tx][ty+1] = s_u2[tx][ty];
		    s_u1lvl[tx][ty+1] = s_u1lvl[tx][ty];
		    s_u2lvl[tx][ty+1] = s_u2lvl[tx][ty];
	    } 
	    else if (threadIdx.y == blockDim.y-1) {
		    s_u1[tx][ty+1] = u_g[idx+pitchf1];
		    s_u2[tx][ty+1] = v_g[idx+pitchf1];
		    s_u1lvl[tx][ty+1] = du_g[idx+pitchf1];
		    s_u2lvl[tx][ty+1] = dv_g[idx+pitchf1];
	    }
	  }

	  __syncthreads();
	  
	//Update Robustifications
	// TODO: rethink this part again
	// shared memory indices 
	unsigned int tx_1 = x == 0 ? tx : tx - 1;
	unsigned int tx1 = x == nx - 1 ? tx : tx + 1;
	unsigned int ty_1 = y == 0 ? ty : ty - 1;
	unsigned int ty1 = y == ny - 1 ? ty : ty + 1;
	
	unsigned int x_1 = x == 0 ? x : x - 1;
	unsigned int x1 = x == nx - 1 ? x : x + 1;
	unsigned int y_1 = y == 0 ? y : y - 1;
	unsigned int y1 = y == ny - 1 ? y : y + 1;

	// global memroy indices. Used to access the texture memory.
	const float xx   = (float)(x) + SF_TEXTURE_OFFSET;
	const float yy   = (float)(y) + SF_TEXTURE_OFFSET;
	const float xx1  = (float)(x1) + SF_TEXTURE_OFFSET;
	const float xx_1 = (float)(x_1) + SF_TEXTURE_OFFSET;
	const float yy1  = (float)(y1) + SF_TEXTURE_OFFSET;
	const float yy_1 = (float)(y_1) + SF_TEXTURE_OFFSET;
	
	// TODO: this part of code is developed under the assumption that _I1pyramid->level[rec_depth][y*nx_fine+x] in cpu code
	// represents the tex2D(tex_flow_sor_I1, xx, yy)
	float Ix = 0.5f*(tex2D(tex_flow_sor_I2, xx1, yy) - tex2D(tex_flow_sor_I2, xx_1, yy) +
					tex2D(tex_flow_sor_I1, xx1, yy)- tex2D(tex_flow_sor_I1, xx_1, yy))*hx;
	float Iy = 0.5f*(tex2D(tex_flow_sor_I2, xx, yy1) - tex2D(tex_flow_sor_I2, xx, yy_1) +
					tex2D(tex_flow_sor_I1, xx, yy1)- tex2D(tex_flow_sor_I1, xx, yy_1))*hy;
	float It = tex2D(tex_flow_sor_I2, xx, yy) - tex2D(tex_flow_sor_I1, xx, yy);
	
	double dxu = (s_u1[tx1][ty] - s_u1[tx_1][ty] + s_u1lvl[tx1][ty] - s_u1lvl[tx_1][ty])*hx;
	double dyu = (s_u1[tx][ty1] - s_u1[tx][ty_1] + s_u1lvl[tx][ty1] - s_u1lvl[tx][ty_1])*hy;
	double dxv = (s_u2[tx1][ty] - s_u2[tx_1][ty] + s_u2lvl[tx1][ty] - s_u2lvl[tx_1][ty])*hx;
	double dyv = (s_u2[tx][ty1] - s_u2[tx][ty_1] + s_u2lvl[tx][ty1] - s_u2lvl[tx][ty_1])*hy;

	double dataterm = s_u1lvl[tx][ty]*Ix + s_u2lvl[tx][ty]*Iy + It;
	// TODO: should I use idx or y*nx+x
	penaltyd_g[y*nx+x] = 1.0f / sqrt(dataterm*dataterm + data_epsilon);
	penaltyr_g[y*nx+x] = 1.0f / sqrt(dxu*dxu + dxv*dxv + dyu*dyu + dyv*dyv + diff_epsilon);
}


/**
 * @brief Precomputes one value as the sum of all values not depending of the
 * current flow increment
 * @param u_g Pointer to global device memory for the horizontal
 * flow component of the accumulation flow field
 * @param v_g Pointer to global device memory for the vertical
 * flow component of the accumulation flow field
 * @param penaltyd_g Pointer to global device memory for data term penalty
 * @param penaltyr_g Pointer to global device memory for regularity term
 * penalty
 * @param bu_g Pointer to global memory for horizontal result value
 * @param bv_g Pointer to global memory for vertical result value
 * @param nx Image width
 * @param ny Image height
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param lambda Smoothness weight
 * @param pitchf1 Image pitch for single float images
 */
__global__ void sorflow_update_righthandside_shared
(
	const float *u_g,
	const float *v_g,
	const float *penaltyd_g,
	const float *penaltyr_g,
	float *bu_g,
	float *bv_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  lambda,
	int    pitchf1
)
{
	// ### Implement Me###
}


/**
 * @brief Kernel to compute one Red-Black-SOR iteration for the nonlinear
 * Euler-Lagrange equation taking into account penalty values and pixel
 * size for warping
 * @param bu_g Right-Hand-Side values for horizontal flow
 * @param bv_g Right-Hand-Side values for vertical flow
 * @param penaltyd_g Pointer to global device memory holding data term penalization
 * @param penaltyr_g Pointer to global device memory holding regularity term
 * penalization
 * @param du_g Pointer to global device memory for the horizontal
 * flow component increment
 * @param dv_g Pointer to global device memory for the vertical
 * flow component increment
 * @param nx Image width
 * @param ny Image height
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param lambda Smoothness weight
 * @param relaxation Overrelaxation for the SOR-solver
 * @param red Parameter deciding whether the red or black fields of a
 * checkerboard pattern are being updated
 * @param pitchf1 Image pitch for single float images
 */
__global__ void sorflow_nonlinear_warp_sor_shared
(
	const float *bu_g,
	const float *bv_g,
	const float *penaltyd_g,
	const float *penaltyr_g,
	float *du_g,
	float *dv_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  lambda,
	float  relaxation,
	int    red,
	int    pitchf1
)
{
	// ### Implement Me ###
}

/**
 * @brief Method that calls the sorflow_nonlinear_warp_sor_shared in a loop,
 * with an outer loop for computing the diffisivity values for
 * one level of a coarse-to-fine implementation.
 * @param u_g Pointer to global device memory for the horizontal
 * flow component
 * @param v_g Pointer to global device memory for the vertical
 * flow component
 * @param du_g Pointer to global device memory for the horizontal
 * flow component increment
 * @param dv_g Pointer to global device memory for the vertical
 * flow component increment
 * @param bu_g Right-Hand-Side values for horizontal flow
 * @param bv_g Right-Hand-Side values for vertical flow
 * @param penaltyd_g Pointer to global device memory holding data term penalization
 * @param penaltyr_g Pointer to global device memory holding regularity term
 * penalization
 * @param nx Image width
 * @param ny Image height
 * @param pitchf1 Image pitch for single float images
 * @param hx Horizontal pixel size
 * @param hy Vertical pixel size
 * @param lambda Smoothness weight
 * @param outer_iterations Number of iterations of the penalty computation
 * @param inner_iterations Number of iterations for the SOR-solver
 * @param relaxation Overrelaxation for the SOR-solver
 * @param data_epsilon Smoothing parameter for the TV Penalization of the data
 * term
 * @param diff_epsilon Smoothing parameter for the TV Penalization of the
 * regularity term
 */
void sorflow_gpu_nonlinear_warp_level
(
		const float *u_g,
		const float *v_g,
		float *du_g,
		float *dv_g,
		float *bu_g,
		float *bv_g,
		float *penaltyd_g,
		float *penaltyr_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float hx,
		float hy,
		float lambda,
		float overrelaxation,
		int   outer_iterations,
		int   inner_iterations,
		float data_epsilon,
		float diff_epsilon
)
{
	// ### Implement Me ###
}


float FlowLibGpuSOR::computeFlow()
{
	// ### Implement Me###
}

