/*
    Linux build command
    g++ -o xva_cpu_stl xva_cpu_stl.cpp -std=c++11 -O3 -lpthread
*/

#include <algorithm>
#include <iostream>
#include <chrono> 
#include <thread>  
#include <mutex>
#include <atomic>

typedef double DataType;
typedef std::tuple<DataType,DataType,DataType,DataType,DataType> DataTypeTuple5;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> HRTime;

int main(int argc, const char* argv[])
{
    for (int threadsCount = 1; threadsCount < 65; threadsCount++)
    {
        int datesCount = 180;
        int simulationsCount = 10000;
        int size = datesCount * simulationsCount;
        
        std::vector<int> h_keys(size);
        for (int i = 0; i < size; i++) 
        {
            h_keys[i] = i % datesCount;
        }
        std::sort(h_keys.begin(), h_keys.end());

        std::vector<int> h_keys_res(datesCount);
        std::vector<DataType> h_cp(size);
        
        std::vector<DataType> h_spot_ccy1(size);
        std::vector<DataType> h_param_ccy1(size);
        std::vector<DataType> h_spot_ccy2(size);
        std::vector<DataType> h_param_ccy2(size);
        
        std::generate(h_cp.begin(), h_cp.end(), rand); 
        std::generate(h_spot_ccy1.begin(), h_spot_ccy1.end(), rand); 
        std::generate(h_param_ccy1.begin(), h_param_ccy1.end(), rand); 
        std::generate(h_spot_ccy2.begin(), h_spot_ccy2.end(), rand); 
        std::generate(h_param_ccy2.begin(), h_param_ccy2.end(), rand); 
        std::generate(h_keys_res.begin(), h_keys_res.end(), rand); 
        
		std::vector<int> executionTimes;
		for (int r = 0; r < 50; r ++) 
		{
			HRTime start = std::chrono::high_resolution_clock::now(); 
			
			
			std::vector<std::thread> threads;			
			std::mutex lock;
			std::atomic_int latch(threadsCount);
			for (int threadId = 0; threadId < threadsCount; threadId++) 
			{
				threads.push_back(std::thread([&, threadId]()
				{
					std::vector<DataType> h_epe(size);
					std::vector<DataType> h_date_epe(datesCount);
					
					std::vector<DataType> d_epe(size);
					std::vector<DataType> d_date_epe(datesCount);
					std::vector<int> d_keys_output(datesCount);
					
					latch--;
					while(latch != 0);
					HRTime algorithmStart = std::chrono::high_resolution_clock::now();
					
					int simIdx = 0;
					int dateIdx = 0;
					for (int k = 0; k < size; k++) 
					{            
						const DataType& cp = h_cp[k];
						const DataType& spot_ccy1 = h_spot_ccy1[k];
						const DataType& param_ccy1 = h_param_ccy1[k];
						const DataType& spot_ccy2 = h_spot_ccy2[k];
						const DataType& param_ccy2 = h_param_ccy2[k];    
						
						DataType res = cp + spot_ccy1 * param_ccy1 - spot_ccy2 - param_ccy2;
						d_epe[simIdx] = std::min(std::max(0.0, res), res * 2.0);
						 
						h_date_epe[dateIdx] += d_epe[simIdx];
						simIdx++;
						if (simIdx % simulationsCount == 0)
							dateIdx++;
					}
					HRTime end = std::chrono::high_resolution_clock::now();
					
					int64_t totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
					int64_t executionTime = std::chrono::duration_cast<std::chrono::microseconds>(end - algorithmStart).count();
						
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

