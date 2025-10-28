#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>

class ThreadPool{
    private:
        bool stop;
        const size_t cores = std::thread::hardware_concurrency();
        std::queue<std::function<void()>> taskQueue;
        std::vector<std::thread> workerThreads;
        std::mutex queueMutex;
        std::condition_variable cV;
        
    public:
        ThreadPool();
        ~ThreadPool();
        void enqueueTask(std::function<void()> task);

};