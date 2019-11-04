/*
    /usr/local/cuda/bin/nvcc -o benchmark benchmark.cu -std=c++11 -O3
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "VariadicTable.h"

typedef double DataType;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> HRTime;

enum Device {
    gpu,
    cpu,
    last
};

enum Algorithm {
    sort,
    sum,
    count_if,
    find_max,
    transform,
    last_alg
};

template <typename T>
struct greater_than_five
{
  __host__ __device__ bool operator()(const T &x) const {return x > 5;}
};

template <typename T>
struct transorm_func
{ 
  __host__ __device__ T operator()(const T &x) const {return x * x + x / 3;}
};

int main(int argc, const char* argv[])
{
    VariadicTable<int, std::string, double, double, double, double, double,  double, double> vt({"Elements", "Algorithm", "HtoD Time, ms", " GPU Execution, ms", "DtoH, ms", "GPU Total Time, ms", "CPU Time, ms", "Diff", "Execution + DtoH Diff"});
    int sizes[] = {1, 1, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 500000000};
    
    for (int iAlgorithm = Algorithm::sort; iAlgorithm != Algorithm::last_alg; iAlgorithm++)
    {   
        for (const int& size : sizes)
        {
            Algorithm algorithm = (Algorithm)iAlgorithm;
            std::string algorithmStr;
             
            HRTime dataTransferStart;        
            HRTime hostTransferStart;
            HRTime hostTransferEnd;

            HRTime gpuAlgorithmStart;
            
            HRTime cpuAlgorithmStart;
            HRTime cpuAlgorithmEnd;
            
            thrust::host_vector<DataType> h_vec(size);
            thrust::host_vector<DataType> h_result_vec(size);
     
            // generate random numbers serially
            std::generate(h_vec.begin(), h_vec.end(), rand); 
            
            for (int iDevice = Device::gpu; iDevice != Device::last; iDevice++)
            {   
                Device device = (Device)iDevice;
                if (device == gpu)
                {
                    // transfer data to the device
                    dataTransferStart = std::chrono::high_resolution_clock::now();
                    thrust::device_vector<DataType> d_vec = h_vec;

                    gpuAlgorithmStart = std::chrono::high_resolution_clock::now();

                    // initial value of the reduction
                    DataType init = 0; 
                    DataType result = 0;
                    switch (algorithm)
                    {
                        case (sort):
                        {
                            algorithmStr = "sort";
                            thrust::sort(d_vec.begin(), d_vec.end());
                            // transfer data back to host
                            hostTransferStart = std::chrono::high_resolution_clock::now();
                            thrust::copy(d_vec.begin(), d_vec.end(), h_result_vec.begin());
                            break;
                        }
                        case (sum):
                        {
                           algorithmStr = "sum";
                            // binary operation used to reduce values
                            thrust::plus<DataType> binary_op;
                            // compute sum on the device
                            result = thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op);
                            hostTransferStart = std::chrono::high_resolution_clock::now();
                            break;
                        }
                        case (count_if):
                        {
                            algorithmStr = "count_if";
                            result = thrust::count_if(d_vec.begin(), d_vec.end(), greater_than_five<DataType>());                    
                            hostTransferStart = std::chrono::high_resolution_clock::now();
                            break;
                        }
                        case (find_max):
                        {
                            algorithmStr = "max";
                            thrust::device_vector<DataType>::iterator iter = thrust::max_element(d_vec.begin(), d_vec.end());
                            result = *iter;
                            hostTransferStart = std::chrono::high_resolution_clock::now();
                            break;
                        }
                        case (transform):
                        {
                            algorithmStr = "transform";
                            thrust::transform(d_vec.begin(),d_vec.end(),d_vec.begin(),transorm_func<DataType>());
                            // transfer data back to host
                            hostTransferStart = std::chrono::high_resolution_clock::now();
                            thrust::copy(d_vec.begin(), d_vec.end(), h_result_vec.begin());
                            break;
                        }
                    }
                    hostTransferEnd = std::chrono::high_resolution_clock::now();
                    std::cout << result << std::endl;
                }
                else if (device == cpu)
                {
                    DataType result = 0;
                    cpuAlgorithmStart = std::chrono::high_resolution_clock::now(); 
                    switch (algorithm)
                    {
                        case (sort):
                        {
                            algorithmStr = "sort";
                            std::sort(h_vec.begin(), h_vec.end());
                            break;
                        }
                        case (sum):
                        {
                            algorithmStr = "sum";
                            result = std::accumulate(h_vec.begin(), h_vec.end(), 0);
                            break;
                        }
                        case (count_if):
                        {
                            algorithmStr = "count_if";
                            result = std::count_if(h_vec.begin(), h_vec.end(), [](DataType i){return i > 5;});
                            break;    
                        }                        
                        case (find_max):
                        {
                            algorithmStr = "max";
                            thrust::host_vector<DataType>::iterator resultIt; 
                            resultIt = std::max_element(h_vec.begin(), h_vec.end());
                            result = *resultIt;
                            break;
                        }
                        case (transform):
                        {
                            algorithmStr = "transform";
                            std::transform(h_vec.begin(), h_vec.end(), h_vec.begin(), transorm_func<DataType>());
                            break;
                        }
                    }
                    cpuAlgorithmEnd = std::chrono::high_resolution_clock::now();
                    std::cout << result << std::endl;
                }
            }
            int64_t hToD = std::chrono::duration_cast<std::chrono::microseconds>(gpuAlgorithmStart - dataTransferStart).count();
            int64_t executionTime = std::chrono::duration_cast<std::chrono::microseconds>(hostTransferStart - gpuAlgorithmStart).count();
            int64_t dToH = std::chrono::duration_cast<std::chrono::microseconds>(hostTransferEnd - hostTransferStart).count();
            int64_t totalTime = std::chrono::duration_cast<std::chrono::microseconds>(hostTransferEnd - dataTransferStart).count();
            int64_t cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(cpuAlgorithmEnd - cpuAlgorithmStart).count();
            double diff = cpuTime / totalTime;       
            double diff2 = cpuTime / (executionTime + dToH);       
            vt.addRow(std::make_tuple(size, algorithmStr, hToD / 1000.0, executionTime / 1000.0, dToH / 1000.0, totalTime / 1000.0, cpuTime / 1000.0, diff, diff2));
        }
    }    
    vt.print(std::cout);
    
    return 0;
}