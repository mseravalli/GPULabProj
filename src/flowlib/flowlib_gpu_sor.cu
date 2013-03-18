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
	// ### Implement Me###
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

  // TODO define the indices and verify correctness

  // current element
  int x = 0;
  int y = 0;

  // element at next position
  unsigned int x_1 = x==0 ? x : x-1;
  unsigned int y_1 = y==0 ? y : y-1;

  // element at previous position
  unsigned int x1 = x==nx-1 ? x : x+1;
  unsigned int y1 = y==ny-1 ? y : y+1;

  unsigned int p = y*nx+x;

  // TODO check validity of declarations
  // ???
  const float hx_1 = 1.0f / (2.0f*hx);
  const float hy_1 = 1.0f / (2.0f*hy);
  const float hx_2 = lambda/(hx*hx);
  const float hy_2 = lambda/(hy*hy);

  // TODO use the texture here
  // calculate the gradient average between 2 resolutions
  float Ix = 0.5f;
  float Iy = 0.5f;
//float Ix = 0.5f*(_I2warp[y*nx+x1]-_I2warp[y*nx+x_1] +
//		_I1pyramid->level[rec_depth][y*nx+x1]-_I1pyramid->level[rec_depth][y*nx+x_1])*hx_1;
//float Iy = 0.5f*(_I2warp[y1*nx+x]-_I2warp[y_1*nx+x] +
//		_I1pyramid->level[rec_depth][y1*nx+x]-_I1pyramid->level[rec_depth][y_1*nx+x])*hy_1;
  
  // p = plus, m = minus
  // average penality of current and previous / current and next
  // why is this needed ???
  float xp = x<nx-1 ? (penaltyr_g[y*nx+x1] +penaltyr_g[y*nx+x])*0.5f*hx_2 : 0.0f;
  float xm = x>0    ? (penaltyr_g[y*nx+x_1]+penaltyr_g[y*nx+x])*0.5f*hx_2 : 0.0f;
  float yp = y<ny-1 ? (penaltyr_g[y1*nx+x] +penaltyr_g[y*nx+x])*0.5f*hy_2 : 0.0f;
  float ym = y>0    ? (penaltyr_g[y_1*nx+x]+penaltyr_g[y*nx+x])*0.5f*hy_2 : 0.0f;
  float sum = xp + xm + yp + ym;
  
  // TODO load d*_g into shared memory????

  // TODO eventually put this if at first pos. in order to reduce computations
  // SOR update adopt Dirichlet boundary conditions
  if((x+y)%2==red){
  	float u1new  = (1.0f-relaxation)*du_g[p] + relaxation *
  			(bu_g[p] - penaltyd_g[p] * Ix*Iy * dv_g[p]
  			+ (x>0    ? xm*du_g[y*nx+x_1] : 0.0f)
  			+ (x<nx-1 ? xp*du_g[y*nx+x1]  : 0.0f)
  			+ (y>0    ? ym*du_g[y_1*nx+x] : 0.0f)
  			+ (y<ny-1 ? yp*du_g[y1*nx+x]  : 0.0f)
  			) / (penaltyd_g[p] * Ix*Ix + sum);
  
  	float u2new = (1.0f-relaxation)*dv_g[p] + relaxation *
  			(bv_g[p] - penaltyd_g[p] * Ix*Iy * du_g[p]
  			+ (x>0    ? xm*dv_g[y*nx+x_1] : 0.0f)
  			+ (x<nx-1 ? xp*dv_g[y*nx+x1]  : 0.0f)
  			+ (y>0    ? ym*dv_g[y_1*nx+x] : 0.0f)
  			+ (y<ny-1 ? yp*dv_g[y1*nx+x]  : 0.0f))
  			/ (penaltyd_g[p] * Iy*Iy + sum);
  	du_g[p] = u1new;
  	dv_g[p] = u2new;
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
  int red = 0;

  // TODO check dimensions: for now it's just copy pasta
  // grid and block dimensions
	int ngx = (nx%SF_BW) ? ((nx/SF_BW)+1) : (nx/SF_BW);
	int ngy = (ny%SF_BH) ? ((ny/SF_BH)+1) : (ny/SF_BH);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);

  for(int i=0; i<outer_iterations ;i++){

    //Update Robustifications
    // sorflow_update_robustifications_warp_tex_shared
    //Update Righthand Side
    // sorflow_update_righthandside_shared

    for(int j=0; j<inner_iterations; j++){
      
      // TODO try to optimize this process: like this half of the processes idle
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


float FlowLibGpuSOR::computeFlow()
{
	// ### Implement Me###
}

