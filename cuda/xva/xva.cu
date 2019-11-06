/*
	Linux build command
	/usr/local/cuda/bin/nvcc -o xva xva.cu -std=c++11 -O3
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <algorithm>
#include <iostream>
#include <chrono> 
#include <thread>  
#include <mutex>
#include <atomic>

typedef double DataType;
typedef thrust::tuple<DataType,DataType,DataType,DataType,DataType> DataTypeTuple5;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> HRTime;

struct transorm_func
{ 
  __host__ __device__ DataType operator()(const DataTypeTuple5 &a) const 
  {
	  const DataType& cp = thrust::get<0>(a);
	  const DataType& spot_ccy1 = thrust::get<1>(a);
	  const DataType& param_ccy1 = thrust::get<2>(a);
	  const DataType& spot_ccy2 = thrust::get<3>(a);
	  const DataType& param_ccy2 = thrust::get<4>(a);

	  DataType res = cp + spot_ccy1 * param_ccy1 - spot_ccy2 - param_ccy2;
	  return fmin(fmax(0, res), res * 2.0);
  }
};

template <typename T1, typename T2>
void host_to_device(T1& hostVec, T2& devVec) 
{
	thrust::copy(hostVec.begin(), hostVec.end(), devVec.begin());
}

template <typename T1, typename T2>
void device_to_host(T1& hostVec, T2& devVec) 
{
	thrust::copy(devVec.begin(), devVec.end(), hostVec.begin());
}

template <typename T1, typename T2>
void host_to_device(cudaStream_t& stream, T1& hostVec, T2& devVec, int elementSize) 
{
	cudaMemcpyAsync(thrust::raw_pointer_cast(devVec.data()), thrust::raw_pointer_cast(hostVec.data()), hostVec.size()*elementSize, cudaMemcpyHostToDevice, stream);
}

template <typename T1, typename T2>
void device_to_host(cudaStream_t& stream, T1& hostVec, T2& devVec, int elementSize) 
{
	cudaMemcpyAsync(thrust::raw_pointer_cast(hostVec.data()), thrust::raw_pointer_cast(devVec.data()), devVec.size()*elementSize, cudaMemcpyDeviceToHost, stream);
}

int main(int argc, const char* argv[])
{
	for (int threadsCount = 1; threadsCount < 65; threadsCount++)
	{
		int datesCount = 180;
		int simulationsCount = 10000;
		int size = datesCount * simulationsCount;
		
		thrust::host_vector<int> h_keys(size);
		for (int i = 0; i < size; i++) 
		{
			h_keys[i] = i % datesCount;
		}
		std::sort(h_keys.begin(), h_keys.end());

		thrust::host_vector<int> h_keys_res(datesCount);
		thrust::host_vector<DataType> h_cp(size);
		
		thrust::host_vector<DataType> h_spot_ccy1(size);
		thrust::host_vector<DataType> h_param_ccy1(size);
		thrust::host_vector<DataType> h_spot_ccy2(size);
		thrust::host_vector<DataType> h_param_ccy2(size);
		
		std::generate(h_cp.begin(), h_cp.end(), rand); 
		std::generate(h_spot_ccy1.begin(), h_spot_ccy1.end(), rand); 
		std::generate(h_param_ccy1.begin(), h_param_ccy1.end(), rand); 
		std::generate(h_spot_ccy2.begin(), h_spot_ccy2.end(), rand); 
		std::generate(h_param_ccy2.begin(), h_param_ccy2.end(), rand); 
		std::generate(h_keys_res.begin(), h_keys_res.end(), rand); 
		
		std::vector<int> executionTimes;
		for (int r = 0; r < 20; r ++) 
		{
			thrust::device_vector<DataType> d_keys(size); 
			thrust::device_vector<DataType> d_cp(size);
			thrust::device_vector<DataType> d_spot_ccy1(size);
			thrust::device_vector<DataType> d_param_ccy1(size);
			thrust::device_vector<DataType> d_spot_ccy2(size);
			thrust::device_vector<DataType> d_param_ccy2(size);
			
			HRTime dataTransferStart = std::chrono::high_resolution_clock::now(); 
			
			host_to_device(h_keys, d_keys);
			host_to_device(h_cp, d_cp);
			host_to_device(h_spot_ccy1, d_spot_ccy1);
			host_to_device(h_param_ccy1, d_param_ccy1);
			host_to_device(h_spot_ccy2, d_spot_ccy2);
			host_to_device(h_param_ccy2, d_param_ccy2);
			
			std::vector<std::thread> threads;
			std::mutex lock;
		
			for (int threadId = 0; threadId < threadsCount; threadId++) 
			{
				std::atomic_int latch(threadsCount);
				threads.push_back(std::thread([&, threadId]()
				{
					cudaStream_t stream;
					cudaStreamCreate(&stream);
					
					thrust::host_vector<DataType> h_epe(size);
					thrust::host_vector<DataType> h_date_epe(datesCount);
					
					thrust::device_vector<DataType> d_epe(size);
					thrust::device_vector<DataType> d_date_epe(datesCount);
					thrust::device_vector<int> d_keys_output(datesCount);
					
					cudaStreamSynchronize(stream);
					
					HRTime gpuAlgorithmStart = std::chrono::high_resolution_clock::now();
					
					host_to_device(stream, h_keys_res, d_keys_output, sizeof(int)); // simulate sending parameters
					
					cudaStreamSynchronize(stream);
					
					auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_cp.begin(), d_spot_ccy1.begin(), d_param_ccy1.begin(), d_spot_ccy2.begin(), d_param_ccy2.begin()));
					auto end = thrust::make_zip_iterator(thrust::make_tuple(d_cp.end(), d_spot_ccy1.end(), d_param_ccy1.end(), d_spot_ccy2.end(), d_param_ccy2.end()));
					thrust::transform(thrust::cuda::par.on(stream),
									  begin, 
									  end, 
									  d_epe.begin(), 
									  transorm_func()); 

					thrust::reduce_by_key(thrust::cuda::par.on(stream), d_keys.begin(), d_keys.end(), d_epe.begin(), d_keys_output.begin(), d_date_epe.begin());

					/* PFE Sorting
					auto keysBegin = thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_epe.begin()));
					auto keysEnd = thrust::make_zip_iterator(thrust::make_tuple(d_keys.end(), d_epe.end()));
					
					thrust::sort_by_key(thrust::cuda::par.on(stream),
										keysBegin,
										keysEnd,
										d_epe.begin(),
										thrust::less<thrust::tuple<int,DataType>>());
					*/   
					HRTime hostTransferStart = std::chrono::high_resolution_clock::now();
					
					device_to_host(stream, h_date_epe, d_date_epe, sizeof(DataType));
					device_to_host(stream, h_keys_res, d_keys_output, sizeof(int));
					
					cudaStreamSynchronize(stream);
					
					HRTime hostTransferEnd = std::chrono::high_resolution_clock::now();
					
					cudaStreamDestroy(stream);
					
					int64_t totalTime = std::chrono::duration_cast<std::chrono::microseconds>(hostTransferEnd - dataTransferStart).count();
					int64_t executionTime = std::chrono::duration_cast<std::chrono::microseconds>(hostTransferEnd - gpuAlgorithmStart).count();
					
					std::lock_guard<std::mutex> guard(lock);
					executionTimes.push_back(executionTime);
				}));
			}
			
			for (auto& thread : threads)
			{
				thread.join();
			}
		}
		int min = *std::min_element(executionTimes.begin(), executionTimes.end());
		int max = *std::max_element(executionTimes.begin(), executionTimes.end());
		double avg = std::accumulate(executionTimes.begin(), executionTimes.end(), 0.0) / executionTimes.size();
		std::sort(executionTimes.begin(), executionTimes.end());
		int median = executionTimes[executionTimes.size() / 2];
		int perc90 = executionTimes[(int)((double)executionTimes.size() * 0.9)];
		printf("Threads count:\t%d Execution time min:\t%d\tmax:\t%d\tavg:\t%.f\tmedian:\t%d\t90%:\t%d\tmicroseconds\n", threadsCount, min, max, avg, median, perc90);
	}
	return 0;
}

