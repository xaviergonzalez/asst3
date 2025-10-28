#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

__global__ void
upsweep_kernel(int N, int two_d, int two_dplus1, int* array){
/* this kernel is used for the upsweep in the pscan
    Args:
        N: length of the array
        two_dplus_1: current receptive field
        array: input array to be processed, device array
*/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // processor index
    int arr_pos = (idx + 1) * two_dplus1 - 1;  // position in the array to be processed
    if (arr_pos < N)
        array[arr_pos] += array[arr_pos - two_d];
}

 __global__ void downsweep_kernel(int N, int two_d, int two_dplus1, int* array) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int arr_pos = (idx + 1) * two_dplus1 - 1;
      if (arr_pos < N) {
          int temp = array[arr_pos - two_d];
          array[arr_pos - two_d] = array[arr_pos];
          array[arr_pos] += temp;
      }
  }

// __global__ void
// downsweep_add(int N, int two_d, int two_dplus1, int* array){
//     /* First parallel op in each step of downsweep*/
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int arr_pos = (idx + 1) * two_dplus1 - 1;
//     if (arr_pos < N)
//         array[arr_pos] += array[arr_pos - two_d];
// }

// __global__ void
// downsweep_set(int N, int two_d, int two_dplus1, int* array){
//     /* Second parallel op in each step of downsweep*/
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int arr_pos = (idx + 1) * two_dplus1 - 1;
//     if (arr_pos < N) {
//        array[arr_pos - two_d] = array[arr_pos] - array[arr_pos - two_d];
//     }
// }

void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    N = nextPow2(N); // round N to next power of 2
    int threads_per_block;
    int threads_needed;
    // if (N < 4194304)
    //     threads_per_block = 256;
    // else
    //     threads_per_block = THREADS_PER_BLOCK;
    // upsweep phase
    for (int two_d = 1; two_d <= N/2; two_d *= 2) {
        int two_dplus1 = two_d * 2;
        threads_needed = N / two_dplus1;
        if (threads_needed < THREADS_PER_BLOCK)
            threads_per_block = threads_needed;
        else
            threads_per_block = THREADS_PER_BLOCK;
        // launch upsweep kernel
        int blocks = CEIL_DIV(threads_needed, threads_per_block);    
        upsweep_kernel<<<blocks, threads_per_block>>>(N, two_d, two_dplus1, result);
        // cudaDeviceSynchronize();
        // dump_device_int_array("up", result, N, two_d, blocks, threads_per_block);
    }

    // set last element to zero
    cudaMemset(result + N - 1, 0, sizeof(int));

    // // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        threads_needed = N / two_dplus1;
        if (threads_needed < THREADS_PER_BLOCK)
            threads_per_block = threads_needed;
        else
            threads_per_block = THREADS_PER_BLOCK;
        int blocks = CEIL_DIV(threads_needed, threads_per_block);
        downsweep_kernel<<<blocks, threads_per_block>>>(N, two_d, two_dplus1,  result);
        // downsweep_add<<<blocks, threads_per_block>>>(N, two_d, two_dplus1, result);
        // dump_device_int_array("d1", result, N, two_d, blocks, threads_per_block);
        // cudaDeviceSynchronize();
        // downsweep_set<<<blocks, threads_per_block>>>(N, two_d, two_dplus1, result);
        // dump_device_int_array("d2", result, N, two_d, blocks, threads_per_block);
    }

}



//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found

__global__ void
compare_shift_left(int length, int* array, int* mask_output, int* val_output){
    /* comparison of adjacent elements in array
    put them into output */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx < length) && (idx > 0)) {
        if (array[idx-1] == array[idx]) {
            // Mark the position for output
            mask_output[idx-1] = 1;
            val_output[idx-1] = 1;
        }
    }
}

__global__ void
scatter_repeat_indices(int length, int* output, int* mask, int* vals){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx < length) && (idx >= 0)) {
        // If the mask indicates a repeat, write the index to the output
        if (mask[idx] == 1) {
            output[vals[idx]] = idx; 
        }
    }

}

void dump_device_int_array(const char* name,
                            const int* dptr,
                            int length){
    int* hptr = (int*)malloc(length * sizeof(int));
    cudaDeviceSynchronize();
    cudaMemcpy(hptr, dptr, length * sizeof(int), cudaMemcpyDeviceToHost);
    fprintf(stderr, "%s:", name);
    for (int i = 0; i < length; i++) {
        fprintf(stderr, "%d ", hptr[i]);
    }
    fprintf(stderr, "\n");
    free(hptr);
}

int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    /*  Args:
        device_input: device array, shape (rounded_length,)
        device_output: device_array, shape (rounded_length,). Will be the mask */
    int* device_mask; // (rounded_length,) device array to hold the mask of repeats
    int* device_vals;  // (rounded_length,) device array to hold the values from the pscan
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_mask, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_vals, rounded_length * sizeof(int));
    // initialize device_output and device_vals to zero
    cudaMemset(device_mask, 0, rounded_length * sizeof(int));
    cudaMemset(device_vals, 0, rounded_length * sizeof(int));
    compare_shift_left<<<CEIL_DIV(length, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(length, device_input, device_mask, device_vals);
    // pscan device_vals to get the indices
    exclusive_scan(device_vals, rounded_length, device_vals);
    scatter_repeat_indices<<<CEIL_DIV(length, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(length, device_output, device_mask, device_vals);
    int output_length = 0;
    cudaMemcpy(&output_length, device_vals + length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    // dump_device_int_array("inpu", device_input, length);
    // dump_device_int_array("mask", device_mask, length);
    // dump_device_int_array("vals", device_vals, length);
    // dump_device_int_array("outp", device_output, length);
    cudaFree(device_mask);
    cudaFree(device_vals);
    return output_length; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
