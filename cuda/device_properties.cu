#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <iomanip>

void printDeviceProperties(const cudaDeviceProp& prop) {
    std::cout << std::left << std::setw(40) << std::setfill(' ') << "Device Name: " << prop.name << std::endl;
    std::cout << std::setw(40) << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::setw(40) << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << std::setw(40) << "Max Blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
    
    // Memory Information
    std::cout << std::setw(40) << "Global Memory: " 
              << prop.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB" << std::endl;
    std::cout << std::setw(40) << "Constant Memory: " << prop.totalConstMem << " bytes" << std::endl;
    std::cout << std::setw(40) << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    
    // Thread and Block Information
    std::cout << std::setw(40) << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << std::setw(40) << "Max Threads Dimension: " 
              << prop.maxThreadsDim[0] << " x " 
              << prop.maxThreadsDim[1] << " x " 
              << prop.maxThreadsDim[2] << std::endl;
    std::cout << std::setw(40) << "Max Grid Size: " 
              << prop.maxGridSize[0] << " x " 
              << prop.maxGridSize[1] << " x " 
              << prop.maxGridSize[2] << std::endl;
    
    // Clock Frequency
    std::cout << std::setw(40) << "Clock Rate: " << prop.clockRate / 1000.0 << " MHz" << std::endl;
    
    // Multiprocessors and Cores
    std::cout << std::setw(40) << "Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
    
    // Registers and Warp
    std::cout << std::setw(40) << "Registers per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    std::cout << std::setw(40) << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << std::setw(40) << "Max Registers per Block: " << prop.regsPerBlock << std::endl;
    
    // Texture and Surface Memory
    std::cout << std::setw(40) << "Max Texture 1D Size: " << prop.maxTexture1D << std::endl;
    std::cout << std::setw(40) << "Max Texture 2D Size: " 
              << prop.maxTexture2D[0] << " x " << prop.maxTexture2D[1] << std::endl;
    
    // Other Features
    std::cout << std::setw(40) << "ECC Support: " 
              << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
    std::cout << std::setw(40) << "Unified Addressing Support: " 
              << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
    std::cout << std::setw(40) << "Compute Mode: ";
    switch (prop.computeMode) {
        case cudaComputeModeDefault:
            std::cout << "Default (Multi-threaded)";
            break;
        case cudaComputeModeExclusive:
            std::cout << "Exclusive";
            break;
        case cudaComputeModeProhibited:
            std::cout << "Prohibited";
            break;
        case cudaComputeModeExclusiveProcess:
            std::cout << "Exclusive Process";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;
    
    // Memory Copy
    std::cout << std::setw(40) << "Async Copy Engines: " << prop.asyncEngineCount << std::endl;
    std::cout << std::setw(40) << "Concurrent Kernels Support: " 
              << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
}

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "Failed to get device count: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "System has " << deviceCount << " CUDA device(s)" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, device);
        
        if (error != cudaSuccess) {
            std::cout << "Failed to get device " << device << " properties: " 
                      << cudaGetErrorString(error) << std::endl;
            continue;
        }
        
        std::cout << "\nDevice " << device << ":" << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        printDeviceProperties(prop);
    }
    
    return 0;
}
