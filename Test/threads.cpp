#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>
#include <pthread.h>
#include <mutex>
#include <semaphore>
#include <functional>

using namespace std;

struct ParForData {
    int from;
    int to;
    function<void(int)> f;
};

#define THREAD_AMOUNT 4

pthread_t threads[THREAD_AMOUNT];
ParForData threadsData[THREAD_AMOUNT];
mutex threadsRunMtx[THREAD_AMOUNT];
mutex threadsDoneMtx[THREAD_AMOUNT];

void* parallel_for_func(void* arg) {
    intptr_t index = reinterpret_cast<intptr_t>(arg);
    threadsDoneMtx[index].unlock();
    ParForData* data = &threadsData[index];
    while(1) {
        threadsRunMtx[index].lock();
        for (int i = data->from; i < data->to; i++) {
            data->f(i);
        }
        threadsDoneMtx[index].unlock();
    }
}

void parallel_for(int n, function<void(int)> f) {
    for(int i = 0; i < THREAD_AMOUNT; i++){
        threadsData[i].from = i * n / THREAD_AMOUNT;
        threadsData[i].to = (i + 1) * n / THREAD_AMOUNT;
        threadsData[i].f = f;
        threadsRunMtx[i].unlock();
    }
    for (int i = 0; i < THREAD_AMOUNT; i++) {
        threadsDoneMtx[i].lock();
    }
}

void init() {
    for(int i = 0; i < THREAD_AMOUNT; i++){
        threadsRunMtx[i].lock();
        threadsDoneMtx[i].lock();
        threadsData[i] = {0, 0};
        pthread_create(&threads[i], nullptr, parallel_for_func, reinterpret_cast<void*>(i));
    }
    for(int i = 0; i < THREAD_AMOUNT; i++){
        threadsDoneMtx[i].lock();
    }
}

int main(){

    init();

    vector<int> nums(100, 0);

    auto f = [&] (int i) {
        nums[i] = i;
    };

    cout << "before 1" << endl;
    parallel_for(100, f);
    cout << "after 1" << endl;

    for(int i = 0; i < 100; i++){
        cout << nums[i] << " ";
    }

    auto g = [&] (int i) {
        nums[i] *= 2;
    };

    cout << "before 2" << endl;
    parallel_for(100, g);
    cout << "after 2" << endl;

    for(int i = 0; i < 100; i++){
        cout << nums[i] << " ";
    }

    return 0;
}