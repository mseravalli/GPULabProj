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
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = y*pitchf1 + x;
	if (x < nx && y < ny) {
		u_g[idx] += du_g[idx];
		v_g[idx] += dv_g[idx];
	}
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
	  const float hx_1 = 1.0f / (2.0f*hx);
	  const float hy_1 = 1.0f / (2.0f*hy);
	  if (x < nx && y < ny) {
		const int tx = threadIdx.x + 1;
		const int ty = threadIdx.y + 1;
		const int idx = y * pitchf1 + x;

		__shared__ float s_u1[SF_BW+2][SF_BH+2];
		__shared__ float s_u2[SF_BW+2][SF_BH+2];
		__shared__ float s_u1lvl[SF_BW+2][SF_BH+2];
		__shared__ float s_u2lvl[SF_BW+2][SF_BH+2];
		  
    // load data into shared memory
		s_u1[tx][ty] = u_g[idx];
		s_u2[tx][ty] = v_g[idx];
		s_u1lvl[tx][ty] = du_g[idx];
		s_u2lvl[tx][ty] = dv_g[idx];

		if (x == 0) {
			s_u1[0][ty] = s_u1[tx][ty];
			s_u2[0][ty] = s_u2[tx][ty];
			s_u1lvl[0][ty] = s_u1lvl[tx][ty];
			s_u2lvl[0][ty] = s_u2lvl[tx][ty];
		} else if (threadIdx.x == 0) {
			s_u1[0][ty] = u_g[idx - 1];
			s_u2[0][ty] = v_g[idx - 1];
			s_u1lvl[0][ty] = du_g[idx - 1];
			s_u2lvl[0][ty] = dv_g[idx - 1];
		}

		if (x == nx - 1) {
			s_u1[tx + 1][ty] = s_u1[tx][ty];
			s_u2[tx + 1][ty] = s_u2[tx][ty];
			s_u1lvl[tx + 1][ty] = s_u1lvl[tx][ty];
			s_u2lvl[tx + 1][ty] = s_u2lvl[tx][ty];
		} else if (threadIdx.x == blockDim.x - 1) {
			s_u1[tx + 1][ty] = u_g[idx + 1];
			s_u2[tx + 1][ty] = v_g[idx + 1];
			s_u1lvl[tx + 1][ty] = du_g[idx + 1];
			s_u2lvl[tx + 1][ty] = dv_g[idx + 1];
		}

		if (y == 0) {
			s_u1[tx][0] = s_u1[tx][ty];
			s_u2[tx][0] = s_u2[tx][ty];
			s_u1lvl[tx][0] = s_u1lvl[tx][ty];
			s_u2lvl[tx][0] = s_u2lvl[tx][ty];
		} else if (threadIdx.y == 0) {
			s_u1[tx][0] = u_g[idx - pitchf1];
			s_u2[tx][0] = v_g[idx - pitchf1];
			s_u1lvl[tx][0] = du_g[idx - pitchf1];
			s_u2lvl[tx][0] = dv_g[idx - pitchf1];
		}

		if (y == ny - 1) {
			s_u1[tx][ty + 1] = s_u1[tx][ty];
			s_u2[tx][ty + 1] = s_u2[tx][ty];
			s_u1lvl[tx][ty + 1] = s_u1lvl[tx][ty];
			s_u2lvl[tx][ty + 1] = s_u2lvl[tx][ty];
		} else if (threadIdx.y == blockDim.y - 1) {
			s_u1[tx][ty + 1] = u_g[idx + pitchf1];
			s_u2[tx][ty + 1] = v_g[idx + pitchf1];
			s_u1lvl[tx][ty + 1] = du_g[idx + pitchf1];
			s_u2lvl[tx][ty + 1] = dv_g[idx + pitchf1];
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
		const float xx   = (float)(x)   + SF_TEXTURE_OFFSET;
		const float yy   = (float)(y)   + SF_TEXTURE_OFFSET;
		const float xx1  = (float)(x1)  + SF_TEXTURE_OFFSET;
		const float xx_1 = (float)(x_1) + SF_TEXTURE_OFFSET;
		const float yy1  = (float)(y1)  + SF_TEXTURE_OFFSET;
		const float yy_1 = (float)(y_1) + SF_TEXTURE_OFFSET;
		
		// TODO: this part of code is developed under the assumption that _I1pyramid->level[rec_depth][y*nx_fine+x] in cpu code
		// represents the tex2D(tex_flow_sor_I1, xx, yy)
		float Ix = 0.5f*(tex2D(tex_flow_sor_I2, xx1, yy) - tex2D(tex_flow_sor_I2, xx_1, yy) +
						tex2D(tex_flow_sor_I1, xx1, yy)- tex2D(tex_flow_sor_I1, xx_1, yy))*hx_1;
		float Iy = 0.5f*(tex2D(tex_flow_sor_I2, xx, yy1) - tex2D(tex_flow_sor_I2, xx, yy_1) +
						tex2D(tex_flow_sor_I1, xx, yy1)- tex2D(tex_flow_sor_I1, xx, yy_1))*hy_1;
		float It = tex2D(tex_flow_sor_I2, xx, yy) - tex2D(tex_flow_sor_I1, xx, yy);
		
		double dxu = (s_u1[tx1][ty] - s_u1[tx_1][ty] + s_u1lvl[tx1][ty] - s_u1lvl[tx_1][ty])*hx_1;
		double dyu = (s_u1[tx][ty1] - s_u1[tx][ty_1] + s_u1lvl[tx][ty1] - s_u1lvl[tx][ty_1])*hy_1;
		double dxv = (s_u2[tx1][ty] - s_u2[tx_1][ty] + s_u2lvl[tx1][ty] - s_u2lvl[tx_1][ty])*hx_1;
		double dyv = (s_u2[tx][ty1] - s_u2[tx][ty_1] + s_u2lvl[tx][ty1] - s_u2lvl[tx][ty_1])*hy_1;
	
		double dataterm = s_u1lvl[tx][ty]*Ix + s_u2lvl[tx][ty]*Iy + It;
		penaltyd_g[idx] = 1.0f / sqrt(dataterm*dataterm + data_epsilon);
		penaltyr_g[idx] = 1.0f / sqrt(dxu*dxu + dxv*dxv + dyu*dyu + dyv*dyv + diff_epsilon);
	  }
}

__global__ void sorflow_update_robustifications_warp_tex
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
  if (!(x < nx && y < ny)) {
    return;
  }

  unsigned int x_1 = x==0     ? x : x-1;
  unsigned int x1 = x==nx-1 ? x : x+1;
  unsigned int y_1 = y==0     ? y : y-1;
  unsigned int y1 = y==ny-1 ? y : y+1;
  
  const float xx   = (float)(x) + SF_TEXTURE_OFFSET;
  const float yy   = (float)(y) + SF_TEXTURE_OFFSET;
  const float xx1  = (float)(x1) + SF_TEXTURE_OFFSET;
  const float xx_1 = (float)(x_1) + SF_TEXTURE_OFFSET;
  const float yy1  = (float)(y1) + SF_TEXTURE_OFFSET;
  const float yy_1 = (float)(y_1) + SF_TEXTURE_OFFSET;

	const float hx_1 = 1.0f / (2.0f*hx);
	const float hy_1 = 1.0f / (2.0f*hy);
    
	float Ix = 0.5f*(tex2D(tex_flow_sor_I2, xx1, yy) - tex2D(tex_flow_sor_I2, xx_1, yy) +
					tex2D(tex_flow_sor_I1, xx1, yy)- tex2D(tex_flow_sor_I1, xx_1, yy))*hx_1;
	float Iy = 0.5f*(tex2D(tex_flow_sor_I2, xx, yy1) - tex2D(tex_flow_sor_I2, xx, yy_1) +
					tex2D(tex_flow_sor_I1, xx, yy1)- tex2D(tex_flow_sor_I1, xx, yy_1))*hy_1;
	float It = tex2D(tex_flow_sor_I2, xx, yy) - tex2D(tex_flow_sor_I1, xx, yy);

  double dxu = (u_g[y *pitchf1+x1] - u_g[y * pitchf1+x_1] + du_g[y* pitchf1+x1] - du_g[y * pitchf1+x_1])*hx_1;
  double dyu = (u_g[y1*pitchf1+x]  - u_g[y_1*pitchf1+x]   + du_g[y1*pitchf1+x]  - du_g[y_1*pitchf1+x]) * hy_1;
  double dxv = (v_g[y *pitchf1+x1] - v_g[y * pitchf1+x_1] + dv_g[y* pitchf1+x1] - dv_g[y * pitchf1+x_1])*hx_1;
  double dyv = (v_g[y1*pitchf1+x]  - v_g[y_1*pitchf1+x]   + dv_g[y1*pitchf1+x]  - dv_g[y_1*pitchf1+x]) * hy_1;
  
  double dataterm = du_g[y*pitchf1+x]*Ix + dv_g[y*pitchf1+x]*Iy + It;
  penaltyd_g[y*pitchf1+x] = 1.0f / sqrt(dataterm*dataterm + data_epsilon);
  penaltyr_g[y*pitchf1+x] = 1.0f / sqrt(dxu*dxu + dxv*dxv + dyu*dyu + dyv*dyv + diff_epsilon);
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
	  const int x = blockIdx.x * blockDim.x + threadIdx.x;
	  const int y = blockIdx.y * blockDim.y + threadIdx.y;
	  const float hx_1 = 1.0f / (2.0f*hx);
	  const float hy_1 = 1.0f / (2.0f*hy);
	  const float hx_2 = lambda/(hx*hx);
	  const float hy_2 = lambda/(hy*hy);
	  
	  if (x < nx && y < ny) {
	 		const int tx = threadIdx.x + 1;
	 		const int ty = threadIdx.y + 1;
	 		const int idx = y * pitchf1 + x;

	 		__shared__ float s_u1[SF_BW+2][SF_BH+2];
	 		__shared__ float s_u2[SF_BW+2][SF_BH+2];
	 		__shared__ float s_penaltyd[SF_BW+2][SF_BH+2];
	 		__shared__ float s_penaltyr[SF_BW+2][SF_BH+2];
	 		  
	 		// load data into shared memory
	 		s_u1[tx][ty] = u_g[idx];
	 		s_u2[tx][ty] = v_g[idx];
	 		s_penaltyd[tx][ty] = penaltyd_g[idx];
	 		s_penaltyr[tx][ty] = penaltyr_g[idx];

	 		if (x == 0) {
	 			s_u1[0][ty] = s_u1[tx][ty];
	 			s_u2[0][ty] = s_u2[tx][ty];
	 			s_penaltyd[0][ty] = s_penaltyd[tx][ty];
	 			s_penaltyr[0][ty] = s_penaltyr[tx][ty];
	 		} else if (threadIdx.x == 0) {
	 			s_u1[0][ty] = u_g[idx - 1];
	 			s_u2[0][ty] = v_g[idx - 1];
	 			s_penaltyd[0][ty] = penaltyd_g[idx - 1];
	 			s_penaltyr[0][ty] = penaltyr_g[idx - 1];
	 		}

	 		if (x == nx - 1) {
	 			s_u1[tx + 1][ty] = s_u1[tx][ty];
	 			s_u2[tx + 1][ty] = s_u2[tx][ty];
	 			s_penaltyd[tx + 1][ty] = s_penaltyd[tx][ty];
	 			s_penaltyr[tx + 1][ty] = s_penaltyr[tx][ty];
	 		} else if (threadIdx.x == blockDim.x - 1) {
	 			s_u1[tx + 1][ty] = u_g[idx + 1];
	 			s_u2[tx + 1][ty] = v_g[idx + 1];
	 			s_penaltyd[tx + 1][ty] = penaltyd_g[idx + 1];
	 			s_penaltyr[tx + 1][ty] = penaltyr_g[idx + 1];
	 		}

	 		if (y == 0) {
	 			s_u1[tx][0] = s_u1[tx][ty];
	 			s_u2[tx][0] = s_u2[tx][ty];
	 			s_penaltyd[tx][0] = s_penaltyd[tx][ty];
	 			s_penaltyr[tx][0] = s_penaltyr[tx][ty];
	 		} else if (threadIdx.y == 0) {
	 			s_u1[tx][0] = u_g[idx - pitchf1];
	 			s_u2[tx][0] = v_g[idx - pitchf1];
	 			s_penaltyd[tx][0] = penaltyd_g[idx - pitchf1];
	 			s_penaltyr[tx][0] = penaltyr_g[idx - pitchf1];
	 		}

	 		if (y == ny - 1) {
	 			s_u1[tx][ty + 1] = s_u1[tx][ty];
	 			s_u2[tx][ty + 1] = s_u2[tx][ty];
	 			s_penaltyd[tx][ty + 1] = s_penaltyd[tx][ty];
	 			s_penaltyr[tx][ty + 1] = s_penaltyr[tx][ty];
	 		} else if (threadIdx.y == blockDim.y - 1) {
	 			s_u1[tx][ty + 1] = u_g[idx + pitchf1];
	 			s_u2[tx][ty + 1] = v_g[idx + pitchf1];
	 			s_penaltyd[tx][ty + 1] = penaltyd_g[idx + pitchf1];
	 			s_penaltyr[tx][ty + 1] = penaltyr_g[idx + pitchf1];
	 		}
	 		
	 		__syncthreads();
	 		
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
			const float xx   = (float)(x)   + SF_TEXTURE_OFFSET;
			const float yy   = (float)(y)   + SF_TEXTURE_OFFSET;
			const float xx1  = (float)(x1)  + SF_TEXTURE_OFFSET;
			const float xx_1 = (float)(x_1) + SF_TEXTURE_OFFSET;
			const float yy1  = (float)(y1)  + SF_TEXTURE_OFFSET;
			const float yy_1 = (float)(y_1) + SF_TEXTURE_OFFSET;
			
			// TODO: this part of code is developed under the assumption that _I1pyramid->level[rec_depth][y*nx_fine+x] in cpu code
			// represents the tex2D(tex_flow_sor_I1, xx, yy)
			float Ix = 0.5f*(tex2D(tex_flow_sor_I2, xx1, yy) - tex2D(tex_flow_sor_I2, xx_1, yy) +
							tex2D(tex_flow_sor_I1, xx1, yy)- tex2D(tex_flow_sor_I1, xx_1, yy))*hx_1;
			float Iy = 0.5f*(tex2D(tex_flow_sor_I2, xx, yy1) - tex2D(tex_flow_sor_I2, xx, yy_1) +
							tex2D(tex_flow_sor_I1, xx, yy1)- tex2D(tex_flow_sor_I1, xx, yy_1))*hy_1;
			float It = tex2D(tex_flow_sor_I2, xx, yy) - tex2D(tex_flow_sor_I1, xx, yy);
			
			float xp = x<nx-1 ? (s_penaltyr[tx1][ty]  + s_penaltyr[tx][ty])*0.5f*hx_2 : 0.0f;
			float xm = x>0    ? (s_penaltyr[tx_1][ty] + s_penaltyr[tx][ty])*0.5f*hx_2 : 0.0f;
			float yp = y<ny-1 ? (s_penaltyr[tx][ty1]  + s_penaltyr[tx][ty])*0.5f*hy_2 : 0.0f;
			float ym = y>0    ? (s_penaltyr[tx][ty_1] + s_penaltyr[tx][ty])*0.5f*hy_2 : 0.0f;
			float sum = xp + xm + yp + ym;
			
			// TODO: rethink the indices of this part again
			bu_g[idx] = -s_penaltyd[tx][ty] * Ix*It
					+ (x>0    ? xm*s_u1[tx_1][ty] : 0.0f)
					+ (x<nx-1 ? xp*s_u1[tx1][ty]  : 0.0f)
					+ (y>0    ? ym*s_u1[tx][ty_1] : 0.0f)
					+ (y<ny-1 ? yp*s_u1[tx][ty1]  : 0.0f)
					- sum * s_u1[tx][ty];

			bv_g[idx] = -s_penaltyd[tx][ty] * Iy*It
					+ (x>0    ? xm*s_u2[tx_1][ty] : 0.0f)
					+ (x<nx-1 ? xp*s_u2[tx1][ty]  : 0.0f)
					+ (y>0    ? ym*s_u2[tx][ty_1] : 0.0f)
					+ (y<ny-1 ? yp*s_u2[tx][ty1]  : 0.0f)
					- sum * s_u2[tx][ty];
	 		  
	  }
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
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const float hx_1 = 1.0f / (2.0f*hx);
  const float hy_1 = 1.0f / (2.0f*hy);
  const float hx_2 = lambda/(hx*hx);
  const float hy_2 = lambda/(hy*hy);
  
  if (x < nx && y < ny) {
		const int tx = threadIdx.x + 1;
		const int ty = threadIdx.y + 1;
		const int idx = y * pitchf1 + x;

		// TODO check how much data can be loaded in shared mem.
		__shared__ float s_bu[SF_BW+2][SF_BH+2];
		__shared__ float s_bv[SF_BW+2][SF_BH+2];
		__shared__ float s_penaltyd[SF_BW+2][SF_BH+2];
		__shared__ float s_penaltyr[SF_BW+2][SF_BH+2];
		__shared__ float s_u1lvl[SF_BW+2][SF_BH+2];
		__shared__ float s_u2lvl[SF_BW+2][SF_BH+2];
		
		// load data into shared memory
		s_bu[tx][ty] = bu_g[idx];
		s_bv[tx][ty] = bv_g[idx];
		s_penaltyd[tx][ty] = penaltyd_g[idx];
		s_penaltyr[tx][ty] = penaltyr_g[idx];
		s_u1lvl[tx][ty] = du_g[idx];
		s_u2lvl[tx][ty] = dv_g[idx];
		
		if (x == 0) {
			s_bu[0][ty] = s_bu[tx][ty];
			s_bv[0][ty] = s_bv[tx][ty];
			s_penaltyd[0][ty] = s_penaltyd[tx][ty];
			s_penaltyr[0][ty] = s_penaltyr[tx][ty];
			s_u1lvl[0][ty] = s_u1lvl[tx][ty];
			s_u2lvl[0][ty] = s_u2lvl[tx][ty];
		} else if (threadIdx.x == 0) {
			s_bu[0][ty] = bu_g[idx - 1];
			s_bv[0][ty] = bv_g[idx - 1];
			s_penaltyd[0][ty] = penaltyd_g[idx - 1];
			s_penaltyr[0][ty] = penaltyr_g[idx - 1];
			s_u1lvl[0][ty] = du_g[idx - 1];
			s_u2lvl[0][ty] = dv_g[idx - 1];
		}

		if (x == nx - 1) {
			s_bu[tx + 1][ty] = s_bu[tx][ty];
			s_bv[tx + 1][ty] = s_bv[tx][ty];
			s_penaltyd[tx + 1][ty] = s_penaltyd[tx][ty];
			s_penaltyr[tx + 1][ty] = s_penaltyr[tx][ty];
			s_u1lvl[tx + 1][ty] = s_u1lvl[tx][ty];
			s_u2lvl[tx + 1][ty] = s_u2lvl[tx][ty];
		} else if (threadIdx.x == blockDim.x - 1) {
			s_bu[tx + 1][ty] = bu_g[idx + 1];
			s_bv[tx + 1][ty] = bv_g[idx + 1];
			s_penaltyd[tx + 1][ty] = penaltyd_g[idx + 1];
			s_penaltyr[tx + 1][ty] = penaltyr_g[idx + 1];
			s_u1lvl[tx + 1][ty] = du_g[idx + 1];
			s_u2lvl[tx + 1][ty] = dv_g[idx + 1];
		}

		if (y == 0) {
			s_bu[tx][0] = s_bu[tx][ty];
			s_bv[tx][0] = s_bv[tx][ty];
			s_penaltyd[tx][0] = s_penaltyd[tx][ty];
			s_penaltyr[tx][0] = s_penaltyr[tx][ty];
			s_u1lvl[tx][0] = s_u1lvl[tx][ty];
			s_u2lvl[tx][0] = s_u2lvl[tx][ty];
		} else if (threadIdx.y == 0) {
			s_bu[tx][0] = bu_g[idx - pitchf1];
			s_bv[tx][0] = bv_g[idx - pitchf1];
			s_penaltyd[tx][0] = penaltyd_g[idx - pitchf1];
			s_penaltyr[tx][0] = penaltyr_g[idx - pitchf1];
			s_u1lvl[tx][0] = du_g[idx - pitchf1];
			s_u2lvl[tx][0] = dv_g[idx - pitchf1];
		}

		if (y == ny - 1) {
			s_bu[tx][ty + 1] = s_bu[tx][ty];
			s_bv[tx][ty + 1] = s_bv[tx][ty];
			s_penaltyd[tx][ty + 1] = s_penaltyd[tx][ty];
			s_penaltyr[tx][ty + 1] = s_penaltyr[tx][ty];
			s_u1lvl[tx][ty + 1] = s_u1lvl[tx][ty];
			s_u2lvl[tx][ty + 1] = s_u2lvl[tx][ty];
		} else if (threadIdx.y == blockDim.y - 1) {
			s_bu[tx][ty + 1] = bu_g[idx + pitchf1];
			s_bv[tx][ty + 1] = bv_g[idx + pitchf1];
			s_penaltyd[tx][ty + 1] = penaltyd_g[idx + pitchf1];
			s_penaltyr[tx][ty + 1] = penaltyr_g[idx + pitchf1];
			s_u1lvl[tx][ty + 1] = du_g[idx + pitchf1];
			s_u2lvl[tx][ty + 1] = dv_g[idx + pitchf1];
		}
		
		__syncthreads();
  
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
		const float xx   = (float)(x)   + SF_TEXTURE_OFFSET;
		const float yy   = (float)(y)   + SF_TEXTURE_OFFSET;
		const float xx1  = (float)(x1)  + SF_TEXTURE_OFFSET;
		const float xx_1 = (float)(x_1) + SF_TEXTURE_OFFSET;
		const float yy1  = (float)(y1)  + SF_TEXTURE_OFFSET;
		const float yy_1 = (float)(y_1) + SF_TEXTURE_OFFSET;
		
		// TODO: this part of code is developed under the assumption that _I1pyramid->level[rec_depth][y*nx_fine+x] in cpu code
		// represents the tex2D(tex_flow_sor_I1, xx, yy)
		float Ix = 0.5f*(tex2D(tex_flow_sor_I2, xx1, yy) - tex2D(tex_flow_sor_I2, xx_1, yy) +	tex2D(tex_flow_sor_I1, xx1, yy)           - tex2D(tex_flow_sor_I1, xx_1, yy))*hx_1;
		float Iy = 0.5f*(tex2D(tex_flow_sor_I2, xx, yy1) - tex2D(tex_flow_sor_I2, xx, yy_1) +	tex2D(tex_flow_sor_I1, xx, yy1)           - tex2D(tex_flow_sor_I1, xx, yy_1))*hy_1;
		
		float xp = x<nx-1 ? (s_penaltyr[tx1][ty] + s_penaltyr[tx][ty])*0.5f*hx_2 : 0.0f;
		float xm = x>0    ? (s_penaltyr[tx_1][ty]+ s_penaltyr[tx][ty])*0.5f*hx_2 : 0.0f;
		float yp = y<ny-1 ? (s_penaltyr[tx][ty1] + s_penaltyr[tx][ty])*0.5f*hy_2 : 0.0f;
		float ym = y>0    ? (s_penaltyr[tx][ty_1]+ s_penaltyr[tx][ty])*0.5f*hy_2 : 0.0f;
		float sum = xp + xm + yp + ym;
		
		//unsigned int p = y*nx_fine+x;
		
		if((x+y)%2==red){
			float u1new  = (1.0f-relaxation)*s_u1lvl[tx][ty] + relaxation *
					(s_bu[tx][ty] - s_penaltyd[tx][ty] * Ix*Iy * s_u2lvl[tx][ty]
					+ (x>0    ? xm*s_u1lvl[tx_1][ty] : 0.0f)
					+ (x<nx-1 ? xp*s_u1lvl[tx1][ty]  : 0.0f)
					+ (y>0    ? ym*s_u1lvl[tx][ty_1] : 0.0f)
					+ (y<ny-1 ? yp*s_u1lvl[tx][ty1]  : 0.0f)
					) / (s_penaltyd[tx][ty] * Ix*Ix + sum);

			float u2new = (1.0f-relaxation)*s_u2lvl[tx][ty] + relaxation *
					(s_bv[tx][ty] - s_penaltyd[tx][ty] * Ix*Iy * s_u1lvl[tx][ty]
					+ (x>0    ? xm*s_u2lvl[tx_1][ty] : 0.0f)
					+ (x<nx-1 ? xp*s_u2lvl[tx1][ty]  : 0.0f)
					+ (y>0    ? ym*s_u2lvl[tx][ty_1] : 0.0f)
					+ (y<ny-1 ? yp*s_u2lvl[tx][ty1]  : 0.0f))
					/ (s_penaltyd[tx][ty] * Iy*Iy + sum);
			du_g[idx] = u1new;
			dv_g[idx] = u2new;
		}
  }
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
  bool red = 0;

  // grid and block dimensions
	int ngx = (nx%SF_BW) ? ((nx/SF_BW)+1) : (nx/SF_BW);
	int ngy = (ny%SF_BH) ? ((ny/SF_BH)+1) : (ny/SF_BH);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);

  for(int i=0; i<outer_iterations ;i++){

    //Update Robustifications
    sorflow_update_robustifications_warp_tex_shared <<<dimGrid,dimBlock>>>
      (	u_g,	v_g,	du_g,	dv_g,	penaltyd_g,	penaltyr_g,	nx,	ny,	hx,	hy,	
        data_epsilon,	diff_epsilon,	pitchf1 );
    //Update Righthand Side
    sorflow_update_righthandside_shared <<<dimGrid,dimBlock>>>
      ( u_g, v_g, penaltyd_g, penaltyr_g, bu_g, bv_g, nx, ny, hx, hy, lambda, 
        pitchf1);
  
    for(int j=0; j<inner_iterations; j++){
      
      // TODO check the effective dimension of the blocks
      red = 0;
      sorflow_nonlinear_warp_sor_shared <<<dimGrid,dimBlock>>>
        ( bu_g, bv_g, penaltyd_g, penaltyr_g, du_g, dv_g, nx, ny, hx, hy, 
          lambda, overrelaxation, red, pitchf1 );
  
      red = 1;
      sorflow_nonlinear_warp_sor_shared <<<dimGrid,dimBlock>>>
        ( bu_g, bv_g, penaltyd_g, penaltyr_g, du_g, dv_g, nx, ny, hx, hy, 
          lambda, overrelaxation, red, pitchf1 );
  
    }

  }

}


float FlowLibGpuSOR::computeFlow() {
	float lambda = _lambda * 255.0f;
	int max_rec_depth;
	int warp_max_levels;
	int rec_depth;

  // grid and block dimensions
  int ngx_glob = (_nx % SF_BW) ? ((_nx / SF_BW) + 1) : (_nx / SF_BW);
  int ngy_glob = (_ny % SF_BH) ? ((_ny / SF_BH) + 1) : (_ny / SF_BH);
  dim3 dimGrid_glob(ngx_glob, ngy_glob);
  dim3 dimBlock_glob(SF_BW, SF_BH);

	warp_max_levels = computeMaxWarpLevels();

	max_rec_depth
			= (((_start_level + 1) < warp_max_levels) ? (_start_level + 1)
					: warp_max_levels) - 1;

	if (max_rec_depth >= _I1pyramid->nl) {
		max_rec_depth = _I1pyramid->nl - 1;
	}

	if (_verbose)
		fprintf(stderr, "\nFlow GPU SOR Relaxation: %f\n", _overrelaxation);

	unsigned int nx_fine, ny_fine, nx_coarse = 0, ny_coarse = 0;

	float hx_fine;
	float hy_fine;

  // set the initial u to 0
  setKernel <<<dimGrid_glob, dimBlock_glob>>> (_u1_g, _nx, _ny, _pitchf1, 0.0f);
  setKernel <<<dimGrid_glob, dimBlock_glob>>> (_u2_g, _nx, _ny, _pitchf1, 0.0f);

	if (_verbose)	fprintf(stderr, "\tu vectors initialized\n");

	for (rec_depth = max_rec_depth; rec_depth >= 0; rec_depth--) {

		if (_verbose) fprintf(stderr, "\tLevel %i\n", rec_depth);

		nx_fine = _I1pyramid->nx[rec_depth];
		ny_fine = _I1pyramid->ny[rec_depth];

		hx_fine = (float) _nx / (float) nx_fine;
		hy_fine = (float) _ny / (float) ny_fine;

    if (_verbose) fprintf(stderr, "\tTexture binding started\n");

        bind_textures(_I1pyramid->level[rec_depth], _I2pyramid->level[rec_depth], 
          nx_fine, ny_fine, _I1pyramid->pitch[rec_depth]);
        textures_flow_sor_initialized = true;

    if (_verbose) fprintf(stderr, "\tTexture binding complete\n");

		if (_debug) {
      _I1 = new float[nx_fine * ny_fine];
      _I2 = new float[nx_fine * ny_fine];

			sprintf(_debugbuffer, "debug/CI1 %i.png", rec_depth);
      cuda_copy_d2h_2D(_I1pyramid->level[rec_depth], _I1,	nx_fine, ny_fine, 1, sizeof(float), _I1pyramid->pitch[rec_depth]);
  		saveFloatImage(_debugbuffer, _I1, nx_fine,	ny_fine, 1, 1.0f, -1.0f);

			sprintf(_debugbuffer, "debug/CI2 %i.png", rec_depth);
      cuda_copy_d2h_2D(_I2pyramid->level[rec_depth], _I2,	nx_fine, ny_fine, 1, sizeof(float), _I2pyramid->pitch[rec_depth]);
  		saveFloatImage(_debugbuffer, _I2, nx_fine,	ny_fine, 1, 1.0f, -1.0f);
      delete [] _I1;
      delete [] _I2;
		}

		if (rec_depth < max_rec_depth) {
      if (_verbose) fprintf(stderr, "\tResampling started\n");

  		resampleAreaParallelSeparate
  		(
  				_u1_g,
  				_u1_g,
  				nx_coarse,
  				ny_coarse,
  				_I2pyramid->pitch[rec_depth+1],
  				nx_fine,
  				ny_fine,
  				_I2pyramid->pitch[rec_depth],
  				_b1
  		);
  		resampleAreaParallelSeparate
  		(
  				_u2_g,
  				_u2_g,
  				nx_coarse,
  				ny_coarse,
  				_I2pyramid->pitch[rec_depth+1],
  				nx_fine,
  				ny_fine,
  				_I2pyramid->pitch[rec_depth],
  				_b2
  		);

      if (_verbose)	fprintf(stderr, "\tResampling complete\n");
		}

    if (rec_depth >= _end_level) {

      // grid and block dimensions
      int ngx = (nx_fine % SF_BW) ? ((nx_fine / SF_BW) + 1) : (nx_fine / SF_BW);
      int ngy = (ny_fine % SF_BH) ? ((ny_fine / SF_BH) + 1) : (ny_fine / SF_BH);
      dim3 dimGrid(ngx, ngy);
      dim3 dimBlock(SF_BW, SF_BH);

      if (_verbose) fprintf(stderr, "\tBack Reg started\n");
      
      backwardRegistrationBilinearFunctionGlobal
      (
          _I2pyramid->level[rec_depth],
          _u1_g,
          _u2_g,
          _I2warp,
          _I1pyramid->level[rec_depth],
          nx_fine,
          ny_fine,
          _I2pyramid->pitch[rec_depth],
          _I1pyramid->pitch[rec_depth],
          hx_fine,
          hy_fine
      );
      
      if (_verbose)	fprintf(stderr, "\tBack Reg complete\n");

      update_textures_flow_sor(_I2warp, nx_fine, ny_fine, _I1pyramid->pitch[rec_depth]);

      if (_debug) {
        _I2 = new float[nx_fine * ny_fine];
        sprintf(_debugbuffer, "debug/CW2 %i.png", rec_depth);
        cuda_copy_d2h_2D(_I2warp, _I2,	nx_fine, ny_fine, 1, sizeof(float), _I2pyramid->pitch[rec_depth]);
        saveFloatImage(_debugbuffer, _I2, nx_fine, ny_fine, 1, 1.0f, -1.0f);
        delete [] _I2;
      }

      // set all derivatives to 0
      setKernel <<<dimGrid,dimBlock>>>(_u1lvl, nx_fine, ny_fine, _I1pyramid->pitch[rec_depth], 0.0f);
      setKernel <<<dimGrid,dimBlock>>>(_u2lvl, nx_fine, ny_fine, _I1pyramid->pitch[rec_depth], 0.0f);

      sorflow_gpu_nonlinear_warp_level(_u1_g, _u2_g, _u1lvl, _u2lvl, _b1,
          _b2, _penDat, _penReg, nx_fine, ny_fine, _I1pyramid->pitch[rec_depth], hx_fine,
          hy_fine, lambda, _overrelaxation, _oi, _ii, _dat_epsilon,
          _reg_epsilon);
      
      // apply the update
      add_flow_fields <<<dimGrid,dimBlock>>>
        (_u1lvl, _u2lvl, _u1_g, _u2_g, nx_fine, ny_fine, _I1pyramid->pitch[rec_depth]);

	}
	else {
		if(_verbose) fprintf(stderr," skipped");
	}
	nx_coarse = nx_fine;
	ny_coarse = ny_fine;

  unbind_textures_flow_sor();
  textures_flow_sor_initialized = false;
}

if(_debug) delete [] _debugbuffer;

//TODO: Timer
return -1.0f;
}

