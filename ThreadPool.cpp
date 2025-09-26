#include "ThreadPool.h"

ThreadPool::ThreadPool(){
    for(size_t i = 0; i < cores; i++){
        workerThreads.emplace_back([this]{
            while(true){
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    //wake up if queue gets a task, or if told to stop
                    cV.wait(lock, [this]{
                        return !taskQueue.empty() || stop;
                    });
                    if(stop && taskQueue.empty()){
                        return;
                    }
                    task = move(taskQueue.front());
                    taskQueue.pop();
                }
                task();
            }
        });
    }
}

void ThreadPool::enqueueTask(std::function<void()> task){
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        taskQueue.emplace(move(task));
    }
    cV.notify_one();
}

ThreadPool::~ThreadPool(){
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    cV.notify_all();
    for (auto& thread : workerThreads) {
        thread.join();
    }
}