/*
 * @brief CUDA implementation for accelerating the Random Mapping Method (RMM).
 *
 * This file provides CUDA kernels and host functions to significantly speed up
 * the most computationally intensive parts of the RMM algorithm. It leverages
 * the GPU for parallel processing of:
 *
 * 1.  Feature Mapping: Utilizes cuBLAS for efficient matrix multiplication
 *     and custom kernels for applying activation functions.
 * 2.  Model Training: Implements the forward pass, gradient calculation, and
 *     parameter updates entirely on the GPU for rapid online learning.
 * 3.  Occupancy Prediction: A highly parallelized kernel for real-time
 *     prediction on large sets of points.
 *
 * The implementation includes optimizations such as stream-based asynchrony,
 * memory pooling, and pinned memory to maximize performance on NVIDIA GPUs.
 */
// random_mapping_cuda.cu

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <Eigen/Dense>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <stdexcept>

// Improved error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(error) << std::endl; \
        throw std::runtime_error(cudaGetErrorString(error)); \
    } \
} while(0)
#define ACTIVATION_SIN 0
#define ACTIVATION_TANH 1
#define ACTIVATION_SIGMOID 2
#define BATCH_SIZE 256  // Optimized for GTX 1650: reduced from 1024 to 256
// Define the optimal thread block size
#define BLOCK_SIZE 256  // Optimized for GTX 1650: 256 is the optimal value
// Error checking macro
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::string error_msg = "CUDA error: "; \
        error_msg += cudaGetErrorString(err); \
        throw std::runtime_error(error_msg); \
    } \
} while(0)
// Add error checking macro
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        cudaDeviceReset(); \
        exit(1); \
    } \
} while(0)
#define CUDA_CHECK_ERROR(msg) do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::string error_msg = msg; \
        error_msg += ": "; \
        error_msg += cudaGetErrorString(err); \
        throw std::runtime_error(error_msg); \
    } \
} while(0)

// Use constant memory to store small parameters (extended for GTX 1650)
__constant__ double c_theta[64];  // Support targetDimen=32
__constant__ double c_bias[64];

// Number of threads in a block, can be adjusted according to actual test
// #define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
/******************************************************************************
 * warpReduceSum: Only applicable to warp size (32) inside the reduction
 *  - Here we use __shfl_down_sync to implement
 ******************************************************************************/
__inline__ __device__ double warpReduceSum(double val) {
    // Typical warp granularity reduction
    // for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/******************************************************************************
 * blockReduceSum: Applicable to a Block (possibly multiple warps)
 *  - First do warp reduction inside the warp, then write the result of each warp to shared memory,
 *    finally do final reduction by a warp.
 ******************************************************************************/
__inline__ __device__ double blockReduceSum(double val) {
    static __shared__ double shared[WARP_SIZE]; // Maximum storage of 32 warp local sums
    int warpId = threadIdx.x / WARP_SIZE;       // Current thread belongs to which warp

    val = warpReduceSum(val);           // First do warp reduction
    __syncthreads();

    if ((threadIdx.x % WARP_SIZE) == 0) {
        // Thread 0 in the warp writes the result to shared
        shared[warpId] = val;
    }
    __syncthreads();

    // Only warp 0 does final reduction again
    double ret = 0.0;
    if (warpId == 0) {
        ret = (threadIdx.x < (blockDim.x / WARP_SIZE)) ? shared[threadIdx.x] : 0.0;
        // Do one more warp reduction
        ret = warpReduceSum(ret);
    }
    return ret;
}

// Error checking macro0127
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::stringstream ss; \
        ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
           << cudaGetErrorString(err); \
        throw std::runtime_error(ss.str()); \
    } \
} while(0)

inline void check_cuda_error(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " 
                  << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// Implementation of activation function and its derivative
__device__ double activate(double x, int activation_type) {
    switch(activation_type) {
        case ACTIVATION_SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case ACTIVATION_SIN:
            return sin(x);
        case ACTIVATION_TANH:
            return tanh(x);
        default:
            return x;
    }
}

__device__ double activate_derivative(double x, int activation_type) {
    switch(activation_type) {
        case ACTIVATION_SIGMOID:
            return x * (1.0 - x);
        case ACTIVATION_SIN:
            return cos(x);
        case ACTIVATION_TANH:
            return 1.0 - x * x;
        default:
            return 1.0;
    }
}

// Auxiliary function: get activation function type
inline int get_activation_type(const char* activation_func) {
    if (strcmp(activation_func, "sigmoid") == 0) {
        return ACTIVATION_SIGMOID;
    } else if (strcmp(activation_func, "tanh") == 0) {
        return ACTIVATION_TANH;
    } else if (strcmp(activation_func, "sin") == 0) {
        return ACTIVATION_SIN;
    }
    return ACTIVATION_SIN;  // Default use sin
}

// Auxiliary function: CUDA memory allocation and check
inline void cudaMalloc_and_check(double** ptr, size_t size) {
    CUDA_CHECK(cudaMalloc((void**)ptr, size * sizeof(double)));
}

// Auxiliary function: memory copy
inline void cuda_memcpy_host_to_device(double* d_ptr, const double* h_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size * sizeof(double), cudaMemcpyHostToDevice));
}

inline void cuda_memcpy_device_to_host(double* h_ptr, const double* d_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size * sizeof(double), cudaMemcpyDeviceToHost));
}

// Performance timer class
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        cudaEventRecord(start_);
    }
    
    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        return milliseconds;
    }

private:
    cudaEvent_t start_, stop_;
};

// Improved memory manager class
class CudaMemoryPool {
public:
    CudaMemoryPool() = default;
    ~CudaMemoryPool() {
        for(auto& pair : pool_) {
            if(pair.second) {
                cudaFree(pair.second);
            }
        }
    }

    double* acquire(size_t size) {
        try {
            if(pool_.find(size) != pool_.end() && pool_[size] != nullptr) {
                double* ptr = pool_[size];
                pool_[size] = nullptr;
                return ptr;
            }
            double* ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&ptr, size));
            return ptr;
        } catch (const std::runtime_error& e) {
            std::cerr << "Failed to acquire memory: " << e.what() << std::endl;
            throw;
        }
    }

    void release(double* ptr, size_t size) {
        try {
            if(pool_[size] != nullptr) {
                CUDA_CHECK(cudaFree(pool_[size]));
            }
            pool_[size] = ptr;
        } catch (const std::runtime_error& e) {
            std::cerr << "Failed to release memory: " << e.what() << std::endl;
            // Try to directly release memory
            if(ptr) {
                cudaFree(ptr);
            }
        }
    }

private:
    std::unordered_map<size_t, double*> pool_;
};

// Optimized kernel
__global__ void optimizedMappingKernel(double* data, const double* bias, 
                                     int m, int n, int actiFunc) {
    extern __shared__ double s_bias[];
    
    if (threadIdx.x < n) {
        s_bias[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < m * n) {
        int col = idx % n;
        double x = data[idx] + s_bias[col];
        
        switch(actiFunc) {
            case 0: // fast sigmoid approximation
                x = 0.5 * (x / (1.0 + abs(x)) + 1.0);
                break;
            case 1: // sin
                x = sin(x);
                break;
            case 2: // linear
                break;
            case 3: // fast tanh approximation
                x = x / (1.0 + abs(x));
                break;
        }
        data[idx] = x;
    }
}

// Configuration struct
struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t sharedMemSize;
};

KernelConfig calculateOptimalConfig(int m, int n) {
    KernelConfig config;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    // int blockSize = std::min(256, maxThreadsPerBlock);
    int blockSize = std::min(BLOCK_SIZE, maxThreadsPerBlock);
    
    int numElements = m * n;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    
    if (gridSize > prop.maxGridSize[0]) {
        gridSize = prop.maxGridSize[0];
    }
    
    config.block = dim3(blockSize);
    config.grid = dim3(gridSize);
    config.sharedMemSize = n * sizeof(double);
    
    return config;
}

// Feature mapping class
class FeatureMapper {
public:
    FeatureMapper() {
        try {
            CUDA_CHECK(cudaStreamCreate(&computeStream_));
            CUDA_CHECK(cudaStreamCreate(&transferStream_));
            
            cublasStatus_t status = cublasCreate(&handle_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cuBLAS handle");
            }
            cublasSetStream(handle_, computeStream_);
        } catch (const std::runtime_error& e) {
            cleanup();
            throw;
        }
    }
    
    ~FeatureMapper() {
        cleanup();
    }
    
    void map(double* h_dataSet, int m, int initial_dim, int n,
             double* h_randomWeights, double* h_randomBias,
             double* h_randomSet, int actiFuncCode, double scaleRate) {
        try {
            CudaTimer timer;
            timer.start();
            
            // Allocate memory
            size_t dataSetSize = m * initial_dim * sizeof(double);
            size_t weightsSize = initial_dim * n * sizeof(double);
            size_t biasSize = n * sizeof(double);
            size_t resultSize = m * n * sizeof(double);
            
            double *d_dataSet = memoryPool_.acquire(dataSetSize);
            double *d_randomWeights = memoryPool_.acquire(weightsSize);
            double *d_randomBias = memoryPool_.acquire(biasSize);
            double *d_randomSetTemp = memoryPool_.acquire(resultSize);
            
            // Asynchronous data transfer
            CUDA_CHECK(cudaMemcpyAsync(d_dataSet, h_dataSet, dataSetSize, 
                                     cudaMemcpyHostToDevice, transferStream_));
            CUDA_CHECK(cudaMemcpyAsync(d_randomWeights, h_randomWeights, weightsSize, 
                                     cudaMemcpyHostToDevice, transferStream_));
            CUDA_CHECK(cudaMemcpyAsync(d_randomBias, h_randomBias, biasSize, 
                                     cudaMemcpyHostToDevice, transferStream_));
            
            CUDA_CHECK(cudaStreamSynchronize(transferStream_));
            
            // Matrix multiplication
            double alpha = scaleRate;
            double beta = 0.0;
            cublasStatus_t status = cublasDgemm(handle_,
                                              CUBLAS_OP_N, CUBLAS_OP_T,
                                              m, n, initial_dim,
                                              &alpha,
                                              d_dataSet, m,
                                              d_randomWeights, n,
                                              &beta,
                                              d_randomSetTemp, m);
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS operation failed");
            }
            
            // Execute mapping kernel
            KernelConfig config = calculateOptimalConfig(m, n);
            optimizedMappingKernel<<<config.grid, config.block, config.sharedMemSize, computeStream_>>>
                (d_randomSetTemp, d_randomBias, m, n, actiFuncCode);
            
            CUDA_CHECK(cudaGetLastError());
            
            // Copy result (asynchronous)
            CUDA_CHECK(cudaMemcpyAsync(h_randomSet, d_randomSetTemp, resultSize, 
                                     cudaMemcpyDeviceToHost, transferStream_));
            
            // GTX 1650 optimization: use stream synchronization instead of global synchronization
            CUDA_CHECK(cudaStreamSynchronize(transferStream_));
            
            // Release memory
            memoryPool_.release(d_dataSet, dataSetSize);
            memoryPool_.release(d_randomWeights, weightsSize);
            memoryPool_.release(d_randomBias, biasSize);
            memoryPool_.release(d_randomSetTemp, resultSize);
            
            float ms = timer.stop();
            // std::cout << "Total execution time: " << ms << " ms" << std::endl;
            
        } catch (const std::runtime_error& e) {
            std::cerr << "Error in feature mapping: " << e.what() << std::endl;
            throw;
        }
    }

private:
    void cleanup() {
        if (computeStream_) {
            cudaStreamDestroy(computeStream_);
        }
        if (transferStream_) {
            cudaStreamDestroy(transferStream_);
        }
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    cudaStream_t computeStream_{};
    cudaStream_t transferStream_{};
    cublasHandle_t handle_{};
    CudaMemoryPool memoryPool_;
};

// External interface
extern "C" void feature_mapping_cuda(
    double* h_dataSet, int m, int initial_dim, int n,
    double* h_randomWeights, double* h_randomBias,
    double* h_randomSet,
    int actiFuncCode, double scaleRate)
{
    try {
        // Input validation
        if (m <= 0 || initial_dim <= 0 || n <= 0) {
            throw std::invalid_argument("Invalid dimensions provided");
        }
        
        if (m > 1000000 || n > 1000000 || initial_dim > 1000000) {
            throw std::invalid_argument("Dimension too large, possible error");
        }
        
        static FeatureMapper mapper;
        mapper.map(h_dataSet, m, initial_dim, n, h_randomWeights, h_randomBias,
                  h_randomSet, actiFuncCode, scaleRate);
                  
    } catch (const std::exception& e) {
        std::cerr << "Error in feature_mapping_cuda: " << e.what() << std::endl;
    }
}

// Modify forward propagation kernel to match model
__global__ void occu_forward_kernel(
    const double* X,
    const double* theta,
    const double* weights,
    const double* bias,
    double* predictions,
    int rows,
    int feature_cols,
    int target_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    double sum = 0.0;
    // Calculate sin term for each dimension
    for (int j = 0; j < target_dim; j++) {
        double wx = 0.0;
        // Calculate w_j * x
        for (int k = 0; k < feature_cols; k++) {
            wx += weights[j * feature_cols + k] * X[idx * feature_cols + k];
        }
        // Calculate theta_j * sin(w_j * x + b_j)
        sum += theta[j] * sin(wx + bias[j]);
    }
    // Add overall bias term theta[target_dim]
    sum += theta[target_dim];
    
    // Numerical truncation: prevent overflow resulting in NaN/Inf
    sum = fmax(fmin(sum, 50.0), -50.0);
    
    // sigmoid activation
    predictions[idx] = 1.0 / (1.0 + exp(-sum));
}

// Modify gradient calculation kernel accordingly
__global__ void occu_gradient_kernel(
    const double* X,
    const double* predictions,
    const double* Y,
    const double* weights,
    const double* bias,
    double* gradient,
    const double* theta,
    int rows,
    int feature_cols,
    int target_dim,
    double lambda
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > target_dim) return;

    double grad_sum = 0.0;
    for (int i = 0; i < rows; i++) {
        double error = predictions[i] - Y[i];
        if (j < target_dim) {
            // Calculate w_j * x
            double wx = 0.0;
            for (int k = 0; k < feature_cols; k++) {
                wx += weights[j * feature_cols + k] * X[i * feature_cols + k];
            }
            // Calculate gradient of sin(w_j * x + b_j)
            grad_sum += error * sin(wx + bias[j]);
        } else {
            // Calculate gradient of overall bias term
            grad_sum += error;
        }
    }
    // Gradient value protection: prevent gradient explosion
    double grad = grad_sum / rows + lambda * theta[j];
    gradient[j] = fmin(fmax(grad, -1e6), 1e6);
}
// Simplified update kernel
__global__ void occu_update_kernel(
    double* theta,
    const double* gradient,
    double* weights,
    double* bias,
    int target_dim,
    int feature_cols,
    double learning_rate
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > target_dim) return;
    theta[j] -= learning_rate * gradient[j];
}

// Training occupancy change rate CUDA implementation using gradient descent
extern "C" void train_occupancy_change_rate_cuda(
    double* h_X,
    double* h_Y,
    double* h_theta,
    double* h_weights,
    double* h_bias,
    int rows,
    int feature_cols,
    int target_dim,
    int num_epochs,
    double learning_rate,
    const char* activation_func
) {
    // Resource management: initialize to nullptr for safe cleanup
    cudaStream_t stream1 = nullptr, stream2 = nullptr;
    double *h_X_pinned = nullptr, *h_Y_pinned = nullptr;
    double *d_X1 = nullptr, *d_Y1 = nullptr, *d_X2 = nullptr, *d_Y2 = nullptr;
    double *d_theta = nullptr, *d_weights = nullptr, *d_bias = nullptr;
    double *d_predictions = nullptr, *d_gradient = nullptr;
    
    try {
        // Optimized for GTX 1650: reduce single training data size to ensure smooth frame rate
        target_dim = min(target_dim, 16);           // Keep 16 dimensions
        const int max_rows = 256;                   // Reduce from 1024 to 256 (key optimization!)
        const int batch_size = min(128, max_rows);  // Reduce from 256 to 128
        rows = min(rows, max_rows);
        
        // Create CUDA stream
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        
        // Allocate fixed memory for asynchronous transmission
        cudaHostAlloc(&h_X_pinned, batch_size * feature_cols * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc(&h_Y_pinned, batch_size * sizeof(double), cudaHostAllocDefault);
        
        // Double buffer allocation
        cudaMalloc(&d_X1, batch_size * feature_cols * sizeof(double));
        cudaMalloc(&d_X2, batch_size * feature_cols * sizeof(double));
        cudaMalloc(&d_Y1, batch_size * sizeof(double));
        cudaMalloc(&d_Y2, batch_size * sizeof(double));
        
        // Model parameter allocation
        cudaMalloc(&d_theta, (target_dim + 1) * sizeof(double));
        cudaMalloc(&d_weights, feature_cols * target_dim * sizeof(double));
        cudaMalloc(&d_bias, target_dim * sizeof(double));
        cudaMalloc(&d_predictions, batch_size * sizeof(double));
        cudaMalloc(&d_gradient, (target_dim + 1) * sizeof(double));

        // Copy model parameters to device
        cudaMemcpyAsync(d_theta, h_theta, (target_dim + 1) * sizeof(double), 
                       cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_weights, h_weights, feature_cols * target_dim * sizeof(double), 
                       cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_bias, h_bias, target_dim * sizeof(double), 
                       cudaMemcpyHostToDevice, stream1);

        // Configure kernel parameters
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid_batch = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGrid_cols = ((target_dim + 1) + threadsPerBlock - 1) / threadsPerBlock;

        // Training loop
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            double current_lr = learning_rate / (1.0 + 0.01 * epoch);
            
            // Batch process
            for (int batch_start = 0; batch_start < rows; batch_start += batch_size) {
                int current_batch_size = min(batch_size, rows - batch_start);
                
                // Prepare batch data
                memcpy(h_X_pinned, h_X + batch_start * feature_cols, 
                      current_batch_size * feature_cols * sizeof(double));
                memcpy(h_Y_pinned, h_Y + batch_start, 
                      current_batch_size * sizeof(double));
                
                // Use double buffer for data transmission
                double *current_d_X = (batch_start / batch_size % 2) ? d_X2 : d_X1;
                double *current_d_Y = (batch_start / batch_size % 2) ? d_Y2 : d_Y1;
                cudaStream_t current_stream = (batch_start / batch_size % 2) ? stream2 : stream1;
                
                // Asynchronous data transmission
                cudaMemcpyAsync(current_d_X, h_X_pinned, 
                              current_batch_size * feature_cols * sizeof(double),
                              cudaMemcpyHostToDevice, current_stream);
                cudaMemcpyAsync(current_d_Y, h_Y_pinned,
                              current_batch_size * sizeof(double),
                              cudaMemcpyHostToDevice, current_stream);
                
                // Execute calculation on corresponding stream
                occu_forward_kernel<<<blocksPerGrid_batch, threadsPerBlock, 0, current_stream>>>(
                    current_d_X, d_theta, d_weights, d_bias, d_predictions,
                    current_batch_size, feature_cols, target_dim
                );
                
                occu_gradient_kernel<<<blocksPerGrid_cols, threadsPerBlock, 0, current_stream>>>(
                    current_d_X, d_predictions, current_d_Y, d_weights, d_bias,
                    d_gradient, d_theta, current_batch_size, feature_cols, target_dim, 0.001
                );
                
                occu_update_kernel<<<blocksPerGrid_cols, threadsPerBlock, 0, current_stream>>>(
                    d_theta, d_gradient, d_weights, d_bias,
                    target_dim, feature_cols, current_lr
                );
            }
            
            // GTX 1650 optimization: further reduce synchronization frequency (key optimization!)
            // Completely remove synchronization during training, only synchronize at the end
            // if (epoch % 500 == 0) {
            //     cudaDeviceSynchronize();
            // }
        }

        // GTX 1650 optimization: asynchronous copy results back to host (reduce blocking)
        cudaMemcpyAsync(h_theta, d_theta, (target_dim + 1) * sizeof(double), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(h_weights, d_weights, feature_cols * target_dim * sizeof(double), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(h_bias, d_bias, target_dim * sizeof(double), cudaMemcpyDeviceToHost, stream1);
        
        // Synchronize all used streams to prevent releasing pinned memory while GPU is still accessing it
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);  // Synchronize stream2 to ensure all asynchronous operations in double buffer are completed

    } catch (const std::exception& e) {
        std::cerr << "Exception in train_occupancy_change_rate_cuda: " << e.what() << std::endl;
    }
    
    // Exception-safe resource cleanup: will execute regardless of whether an exception is thrown
    if (h_X_pinned) cudaFreeHost(h_X_pinned);
    if (h_Y_pinned) cudaFreeHost(h_Y_pinned);
    if (stream1) cudaStreamDestroy(stream1);
    if (stream2) cudaStreamDestroy(stream2);
    
    if (d_X1) cudaFree(d_X1);
    if (d_X2) cudaFree(d_X2);
    if (d_Y1) cudaFree(d_Y1);
    if (d_Y2) cudaFree(d_Y2);
    if (d_theta) cudaFree(d_theta);
    if (d_weights) cudaFree(d_weights);
    if (d_bias) cudaFree(d_bias);
    if (d_predictions) cudaFree(d_predictions);
    if (d_gradient) cudaFree(d_gradient);
}

// Optimized kernel implementation, add boundary check
__global__ void predict_occu_change_rate_kernel(
    const double* __restrict__ X,
    const double* __restrict__ theta,
    const double* __restrict__ weights,
    const double* __restrict__ bias,
    double* __restrict__ predictions,
    int rows,
    int feature_cols,
    int target_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    // Use registers to cache, limit size
    const int MAX_TARGET_DIM = 32;
    double sum = 0.0;
    double feature_sum = 0.0;
    
    // Safety check
    if (target_dim > MAX_TARGET_DIM) return;
    
    // Calculate dimension by dimension to avoid array out of bounds
    for (int j = 0; j < target_dim; j++) {
        feature_sum = 0.0;
        
        // Calculate feature sum
        for (int k = 0; k < feature_cols; k++) {
            feature_sum += X[idx * feature_cols + k] * weights[k * target_dim + j];
        }
        
        feature_sum += bias[j];
        sum += theta[j] * sin(feature_sum);
    }
    
    sum += theta[target_dim];
    
    // Numerical truncation: prevent overflow resulting in NaN/Inf
    sum = fmax(fmin(sum, 50.0), -50.0);
    
    predictions[idx] = 1.0 / (1.0 + exp(-sum));
}

extern "C" void predict_occupancy_change_rate_cuda(
    const double* X_normalized,
    const double* theta,
    const double* weights,
    const double* bias,
    double* Y_predicted,
    int rows,
    int feature_cols,
    int target_dim,
    const char* activation_func
) {
    // Resource management: initialize to nullptr for safe cleanup
    cudaStream_t stream = nullptr;
    double *h_X_pinned = nullptr, *h_Y_pinned = nullptr;
    double *d_X = nullptr, *d_Y = nullptr;
    double *d_theta = nullptr, *d_weights = nullptr, *d_bias = nullptr;
    
    try {
        // Basic parameter check
        if (!X_normalized || !theta || !weights || !bias || !Y_predicted) {
            throw std::runtime_error("Null pointer passed as argument");
        }

        if (rows <= 0 || feature_cols <= 0 || target_dim <= 0) {
            throw std::runtime_error("Invalid dimensions");
        }

        // Optimized for GTX 1650: moderate batch_size to ensure smooth frame time
        const int batch_size = std::min(256, rows);  // Reduce from 512 to 256, smoother frame time
        const int num_batches = (rows + batch_size - 1) / batch_size;

        // Create CUDA stream for asynchronous operation
        CUDA_SAFE_CALL(cudaStreamCreate(&stream));

        // Allocate pinned memory for asynchronous transmission
        CUDA_SAFE_CALL(cudaHostAlloc(&h_X_pinned, batch_size * feature_cols * sizeof(double), 
                                     cudaHostAllocDefault));
        CUDA_SAFE_CALL(cudaHostAlloc(&h_Y_pinned, batch_size * sizeof(double), 
                                     cudaHostAllocDefault));

        // Allocate device memory
        CUDA_SAFE_CALL(cudaMalloc(&d_X, batch_size * feature_cols * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc(&d_Y, batch_size * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc(&d_theta, (target_dim + 1) * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc(&d_weights, feature_cols * target_dim * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc(&d_bias, target_dim * sizeof(double)));

        // Copy model parameters to device (asynchronous)
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_theta, theta, (target_dim + 1) * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_weights, weights, feature_cols * target_dim * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_bias, bias, target_dim * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

        // Configure kernel parameters: optimized for GTX 1650
        const int blockSize = 256;  // Optimal value for GTX 1650
        const int numBlocks = (batch_size + blockSize - 1) / blockSize;

        // Batch process (asynchronous pipeline)
        for (int i = 0; i < num_batches; i++) {
            int current_batch_size = std::min(batch_size, rows - i * batch_size);
            if (current_batch_size <= 0) break;

            // Prepare data to pinned memory
            memcpy(h_X_pinned, X_normalized + i * batch_size * feature_cols,
                   current_batch_size * feature_cols * sizeof(double));

            // Asynchronous copy input data
            CUDA_SAFE_CALL(cudaMemcpyAsync(d_X, h_X_pinned,
                                          current_batch_size * feature_cols * sizeof(double),
                                          cudaMemcpyHostToDevice, stream));

            // Execute kernel
            predict_occu_change_rate_kernel<<<numBlocks, blockSize, 0, stream>>>(
                d_X, d_theta, d_weights, d_bias, d_Y,
                current_batch_size, feature_cols, target_dim
            );

            // Asynchronous copy results
            CUDA_SAFE_CALL(cudaMemcpyAsync(h_Y_pinned, d_Y,
                                          current_batch_size * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream));

            // Wait for current batch to complete
            CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

            // Copy to final output
            memcpy(Y_predicted + i * batch_size, h_Y_pinned,
                   current_batch_size * sizeof(double));
        }

        // Synchronize stream to ensure all operations are completed
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

    } catch (const std::exception& e) {
        std::cerr << "CUDA Error in predict_occupancy_change_rate_cuda: " 
                  << e.what() << std::endl;
    }
    
    // Exception-safe resource cleanup: will execute regardless of whether an exception is thrown
    if (stream) cudaStreamDestroy(stream);
    if (h_X_pinned) cudaFreeHost(h_X_pinned);
    if (h_Y_pinned) cudaFreeHost(h_Y_pinned);
    if (d_X) cudaFree(d_X);
    if (d_Y) cudaFree(d_Y);
    if (d_theta) cudaFree(d_theta);
    if (d_weights) cudaFree(d_weights);
    if (d_bias) cudaFree(d_bias);
}