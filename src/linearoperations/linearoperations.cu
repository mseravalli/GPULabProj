/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: superresolution
* file:    linearoperations.cu
*
*
* implement all functions with ### implement me ### in the function body
\****************************************************************************/

/*
 * linearoperations.cu
 *
 *  Created on: Aug 3, 2012
 *      Author: steinbrf
 */


#include <auxiliary/cuda_basic.cuh>

#include <math.h>

cudaChannelFormatDesc linearoperation_float_tex = cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_linearoperation;
bool linearoperation_textures_initialized = false;


#define MAXKERNELRADIUS     20    // maximum allowed kernel radius
#define MAXKERNELSIZE   21    // maximum allowed kernel radius + 1
__constant__ float constKernel[MAXKERNELSIZE];


void setTexturesLinearOperations(int mode){
	tex_linearoperation.addressMode[0] = cudaAddressModeClamp;
	tex_linearoperation.addressMode[1] = cudaAddressModeClamp;
	if(mode == 0)tex_linearoperation.filterMode = cudaFilterModePoint;
	else tex_linearoperation.filterMode = cudaFilterModeLinear;
	tex_linearoperation.normalized = false;
}


#define LO_TEXTURE_OFFSET 0.5f
#define LO_RS_AREA_OFFSET 0.0f

#ifdef DGT400
#define LO_BW 32
#define LO_BH 16
#else
#define LO_BW 16
#define LO_BH 16
#endif


#ifndef RESAMPLE_EPSILON
#define RESAMPLE_EPSILON 0.005f
#endif

#ifndef atomicAdd
__device__ float atomicAdd(float* address, double val)
{
	unsigned int* address_as_ull = (unsigned int*)address;
	unsigned int old = *address_as_ull, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__float_as_int(val + __int_as_float(assumed)));
	}	while (assumed != old);
	return __int_as_float(old);
}

#endif


void backwardRegistrationBilinearValueTex
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		float value,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
)
{
	// ### Implement me
  fprintf(stderr, "\nERROR this function is not yet implemented\n");
}

__global__ void backwardRegistrationBilinearFunctionGlobalGpu
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		const float *constant_g,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// check if x is within the boundaries
	if (!(x < nx && y < ny)) {
		return;
	}

	const float xx = (float) x + flow1_g[y * pitchf1_in + x] / hx;
	const float yy = (float) y + flow2_g[y * pitchf1_in + x] / hy;

	int xxFloor = (int) floor(xx);
	int yyFloor = (int) floor(yy);

	int xxCeil = xxFloor == nx - 1 ? xxFloor : xxFloor + 1;
	int yyCeil = yyFloor == ny - 1 ? yyFloor : yyFloor + 1;

	float xxRest = xx - (float) xxFloor;
	float yyRest = yy - (float) yyFloor;

	out_g[y * pitchf1_out + x] = (xx < 0.0f || yy < 0.0f || xx > (float) (nx
			- 1) || yy > (float) (ny - 1)) ? constant_g[y * pitchf1_in + x]
			: (1.0f - xxRest) * (1.0f - yyRest) * in_g[yyFloor * pitchf1_in
					+ xxFloor] + xxRest * (1.0f - yyRest) * in_g[yyFloor
					* pitchf1_in + xxCeil] + (1.0f - xxRest) * yyRest
					* in_g[yyCeil * pitchf1_in + xxFloor] + xxRest * yyRest
					* in_g[yyCeil * pitchf1_in + xxCeil];

	  
}

void backwardRegistrationBilinearFunctionGlobal
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		const float *constant_g,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
)
{
	
	  int ngx = (nx%LO_BW) ? ((nx/LO_BW)+1) : (nx/LO_BW);
	  int ngy = (ny%LO_BH) ? ((ny/LO_BH)+1) : (ny/LO_BH);
	  dim3 dimGrid(ngx,ngy);
	  dim3 dimBlock(LO_BW,LO_BH);
	  
		backwardRegistrationBilinearFunctionGlobalGpu<<< dimGrid,dimBlock >>>
		(
				in_g,
				flow1_g,
				flow2_g,
				out_g,
				constant_g,
				nx,
				ny,
				pitchf1_in,
				pitchf1_out,
				hx,
				hy
		);
	  
}

void backwardRegistrationBilinearFunctionTex
(
		const float *in_g,
		const float *flow1_g,
		const float *flow2_g,
		float *out_g,
		const float *constant_g,
		int   nx,
		int   ny,
		int   pitchf1_in,
		int   pitchf1_out,
		float hx,
		float hy
)
{
	// ### Implement me, if you want ###
  fprintf(stderr, "\nERROR this function is not yet implemented\n");
}




void forewardRegistrationBilinearAtomic
(
		const float *flow1_g,
		const float *flow2_g,
		const float *in_g,
	  float       *out_g,
		int         nx,
		int         ny,
		int         pitchf1
)
{
	// ### Implement me ###
  fprintf(stderr, "\nERROR this function is not yet implemented\n");
}




void gaussBlurSeparateMirrorGpu
(
		float *in_g,
		float *out_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float sigmax,
		float sigmay,
		int   radius,
		float *temp_g,
		float *mask
)
{
	// ### Implement me ###
  fprintf(stderr, "\nERROR this function is not yet implemented\n");
}

__global__ void resampleAreaParallelSeparateGpu_x
(
		const float *in_g,
		float *out_g,
		int   nx,
		int   ny,
		float hx,
		int   pitchf1_in,
		int   pitchf1_out,
		float scalefactor = 0.0f
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// check if x is within the boundaries
	if (!(x < nx && y < ny)) {
		return;
	}

	int p = y * pitchf1_out + x;
	// resampling in x
	if (scalefactor == 0.0f)
		scalefactor = 1.0f / hx;

	float px = (float) x * hx;
	float left = ceil(px) - px;
	if (left > hx)
		left = hx;
	float midx = hx - left;
	float right = midx - floorf(midx);
	midx = midx - right;

	out_g[p] = 0.0f;

	if (left > 0.0f) {
		out_g[p] += in_g[y * pitchf1_in + (int) (floor(px))] * left * scalefactor;
		px += 1.0f;
	}
	while (midx > 0.0f) {
		out_g[p] += in_g[y * pitchf1_in + (int) (floor(px))] * scalefactor;
		px += 1.0f;
		midx -= 1.0f;
	}
	if (right > RESAMPLE_EPSILON) {
		out_g[p] += in_g[y * pitchf1_in + (int) (floor(px))] * right
				* scalefactor;
	}

}

__global__ void resampleAreaParallelSeparateGpu_y
(
		const float *in_g,
		float *out_g,
		int   nx,
		int   ny,
		float hy,
		int   pitchf1,
		float scalefactor = 0.0f
) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// check if x is within the boundaries
	if (!(x < nx && y < ny)) {
		return;
	}

	int p = y * pitchf1 + x;

	// resampling in y
	if (scalefactor == 0.0f)
		scalefactor = 1.0f / hy;

	float py = (float) y * hy;
	float top = ceil(py) - py;
	if (top > hy)
		top = hy;
	float midy = hy - top;
	float bottom = midy - floorf(midy);
	midy = midy - bottom;

	out_g[p] = 0.0f;

	if (top > 0.0f) {
		out_g[p] += in_g[(int) (floor(py)) * pitchf1 + x] * top * scalefactor;
		py += 1.0f;
	}
	while (midy > 0.0f) {
		out_g[p] += in_g[(int) (floor(py)) * pitchf1 + x] * scalefactor;
		py += 1.0f;
		midy -= 1.0f;
	}
	if (bottom > RESAMPLE_EPSILON) {
		out_g[p] += in_g[(int) (floor(py)) * pitchf1 + x] * bottom * scalefactor;
	}
	
}

void resampleAreaParallelSeparate
(
		const float *in_g,
		float *out_g,
		int   nx_in,
		int   ny_in,
		int   pitchf1_in,
		int   nx_out,
		int   ny_out,
		int   pitchf1_out,
		float *help_g,
		float scalefactor = 1.0f
)
{
	bool selfalloc = help_g == 0;
	if (selfalloc) {
		fprintf(stderr, "\nADVICE: Use a helper array for separate Resampling!");
		size_t iPitchBytes = 0;
		cutilSafeCall(
				cudaMallocPitch((void**) &(help_g), &iPitchBytes,
						max(nx_in, nx_out) * sizeof(float), max(ny_in, ny_out)));
		pitchf1_out = iPitchBytes / sizeof(float);
	}
	float hx = (float) (nx_in) / (float) (nx_out);
	float hy = (float) (ny_in) / (float) (ny_out);

	int ngx_resx = (nx_out % LO_BW) ? ((nx_out / LO_BW) + 1) : (nx_out / LO_BW);
	int ngy_resx = (ny_in % LO_BH) ? ((ny_in / LO_BH) + 1) : (ny_in / LO_BH);
	dim3 dimGrid_resx(ngx_resx, ngy_resx);
	dim3 dimBlock_resx(LO_BW, LO_BH);

  resampleAreaParallelSeparateGpu_x <<< dimGrid_resx,dimBlock_resx >>>
		  (
		  		in_g,
		  		help_g,
		  		nx_out,
		  		ny_in,
		  		hx,
		  		pitchf1_in,
		  		pitchf1_out,
		  		(float)(nx_out)/(float)(nx_in) 
		  );

	int ngx_resy = (nx_out % LO_BW) ? ((nx_out / LO_BW) + 1) : (nx_out / LO_BW);
	int ngy_resy = (ny_out % LO_BH) ? ((ny_out / LO_BH) + 1) : (ny_out / LO_BH);
	dim3 dimGrid_resy(ngx_resy, ngy_resy);
	dim3 dimBlock_resy(LO_BW, LO_BH);
  resampleAreaParallelSeparateGpu_y <<< dimGrid_resy,dimBlock_resy >>>
		  (
				help_g,
		  		out_g,
		  		nx_out,
		  		ny_out,
		  		hy,
		  		pitchf1_out,
		  		scalefactor*(float)(ny_out)/(float)(ny_in) 
		  );
}

void resampleAreaParallelSeparateAdjoined
(
		const float *in_g,
		float *out_g,
		int   nx_in,
		int   ny_in,
		int   pitchf1_in,
		int   nx_out,
		int   ny_out,
		int   pitchf1_out,
		float *help_g,
		float scalefactor
)
{
	// ### Implement me ###
  fprintf(stderr, "\nERROR this function is not yet implemented\n");
}


__global__ void addKernel
(
		const float *increment_g,
		float *accumulator_g,
		int   nx,
		int   ny,
		int   pitchf1
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = y*pitchf1 + x;
	if (x < nx && y < ny) {
		accumulator_g[idx] += increment_g[idx];
	}
}

__global__ void subKernel
(
		const float *increment_g,
		float *accumulator_g,
		int   nx,
		int   ny,
		int   pitchf1
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = y*pitchf1 + x;
	if (x < nx && y < ny) {
		accumulator_g[idx] -= increment_g[idx];
	}
}

__global__ void setKernel
(
		float *field_g,
		int   nx,
		int   ny,
		int   pitchf1,
		float value
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = y*pitchf1 + x;
	if (x < nx && y < ny) {
		field_g[idx] = value;
	}
}

