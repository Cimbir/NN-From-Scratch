#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>
#include <thread>
#include <mutex>
#include <functional>
#include <iomanip>

using namespace std;

// ================== Global Variables ==================

// Neural Network Architecture
#define Vec double*
#define Mat Vec*
#define Net Mat*

// Threading
#define THREAD_AMOUNT 2

#define PARALLEL_DELTA_UPDATE 0
#define PARALLEL_WEIGHT_UPDATE 0

struct ParForData {
    int from;
    int to;
    function<void(int)> f;
};

thread threads[THREAD_AMOUNT];
ParForData threadsData[THREAD_AMOUNT];
mutex threadsRunMtx[THREAD_AMOUNT];
mutex threadsDoneMtx[THREAD_AMOUNT];

// Data
#define Data_Entry pair<Vec, Vec>

// Neural Network Structure
int layer_n = 4;
int layer_sz[] = {2,20,20,20,2};

Net weights;

Vec* beforeActivation;
Vec* afterActivation;
Vec* delta;

// Activation

#define _relu 1
#define _sigmoid 0
#define _activation 0 

// ================== Function Definitions ==================

// Utils
string to_string(Vec& v);
string to_string(Mat& m);
string to_string(Net& n);

// Parallel For Loop
void* parallel_for_func(void* arg);
void parallel_for(int n, function<void(int)> f);
void init_parallel_for();

// Activation Functions
double relu(double x);
double relu_d(double x, double y);
double sigmoid(double x);
double sigmoid_d(double x, double y);
double activation(double x);
double activation_d(double x, double y);

// Neural Network Functions
Vec add_bias(Vec v);
Vec forward(Vec input);
void backward(Vec input, Vec result, double lr);
void train(Data_Entry* dataset, int n, int epochs, double& lr);
void init_network();

// Setup
void init();

// Data
Data_Entry* getCircleData(int n, int w, int h, double x, double y, double r);

// ================== Utils ==================

string to_string(Vec v, int n){
    string res = "[";
    for(int i = 0; i < n; i++){
        //res += to_string(v[i]).substr(0, to_string(v[i]).find(".") + 3);
        res += to_string(v[i]);
        if(i < n - 1){
            res += ", ";
        }
    }
    res += "]";
    return res;
}

string to_string(Mat ma, int n, int m){
    string res = "[";
    for(int i = 0; i < n; i++){
        res += to_string((Vec)ma[i],m);
        if(i < n - 1){
            res += ",\n";
        }
    }
    res += "]";
    return res;
}

string to_string(Net n, int* sizes, int am){
    string res = "[";
    for(int i = 0; i < am; i++){
        res += to_string(n[i], sizes[i+1], sizes[i]);
        if(i < am - 1){
            res += ",\n";
        }
    }
    res += "]";
    return res;
}

// ================== Parallel For Loop ==================

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

void init_parallel_for() {
    for(int i = 0; i < THREAD_AMOUNT; i++){
        threadsRunMtx[i].lock();
        threadsDoneMtx[i].lock();
        threadsData[i] = {0, 0};
        threads[i] = thread(parallel_for_func, reinterpret_cast<void*>(i));
    }
    for(int i = 0; i < THREAD_AMOUNT; i++){
        threadsDoneMtx[i].lock();
    }
}

// ================== Activation Functions ==================

double relu(double x){
    return max(0.0, x);
}

double relu_d(double x, double y){
    return x > 0 ? 1 : 0;
}

double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

double sigmoid_d(double x, double y){
    return y * (1 - y);
}

double activation(double x){
    if(_activation == _relu) return relu(x);
    if(_activation == _sigmoid) return sigmoid(x);
    return 0;
}

double activation_d(double x, double y){
    if(_activation == _relu) return relu_d(x, y);
    if(_activation == _sigmoid) return sigmoid_d(x, y);
    return 0;
}

// ================== Neural Network Functions ==================

Vec add_bias(Vec v, int n){
    Vec res = new double[n+1];
    for(int i = 0; i < n; i++){
        res[i] = v[i];
    }
    res[n] = 1;
    return res;
}

Vec forward(Vec input) {
    input = add_bias(input, layer_sz[0]);
    for(int i = 0; i < layer_n; i++){
        for(int j = 0; j < layer_sz[i+1]; j++){
            double sum = 0;
            for(int k = 0; k < layer_sz[i]; k++){
                double f = weights[i][j][k];
                double s = input[k];
                sum += f * s;
            }
            beforeActivation[i][j] = sum;
            afterActivation[i][j] = activation(sum);
        }
        input = afterActivation[i];
    }
    return input;
}

void backward(Vec input, Vec result, double lr = 1) {
    Vec output = forward(input);
    input = add_bias(input, layer_sz[0]);

    // Update deltas
    for(int i = 0; i < layer_sz[layer_n]; i++){
        delta[layer_n-1][i] = activation_d(beforeActivation[layer_n-1][i], afterActivation[layer_n-1][i]) * (result[i] - output[i]);
    }
    for(int i = layer_n-2; i >= 0; i--){
        for(int j = 0; j < layer_sz[i+1]; j++){
            double sum = 0;
            for(int k = 0; k < layer_sz[i+2]; k++){
                sum += delta[i+1][k] * weights[i+1][k][j];
            }
            delta[i][j] = activation_d(beforeActivation[i][j], afterActivation[i][j]) * sum;
        }
    }

    // Update weights
    for(int i = 0; i < layer_n; i++){
        for(int j = 0; j < layer_sz[i+1]; j++){
            for(int k = 0; k < layer_sz[i]; k++){
                if(i == 0){
                    weights[i][j][k] += lr * delta[i][j] * input[k];
                }else{
                    weights[i][j][k] += lr * delta[i][j] * afterActivation[i-1][k];
                }
            }
        }
    }
}

void train(Data_Entry* dataset, int n, int epochs, double& lr) {
    for(int i = 0; i < epochs; i++){
        for(int j = 0; j < n; j++){
            backward(dataset[j].first, dataset[j].second, lr);
        }
        if(i % 50 == 0) lr *= 0.99;
    }
}

void init_network() {
    weights = new Mat[layer_n];

    beforeActivation = new Vec[layer_n];
    afterActivation = new Vec[layer_n];
    delta = new Vec[layer_n];

    layer_sz[0]++; // For bias
    for(int i = 0; i < layer_n; i++){
        // Weights
        weights[i] = new Vec[layer_sz[i+1]];
        for(int j = 0; j < layer_sz[i+1]; j++){
            weights[i][j] = new double[layer_sz[i]];
            for(int k = 0; k < layer_sz[i]; k++){
                weights[i][j][k] = (rand() % 100) / 100.0;
            }
        }

        // Before Activation
        beforeActivation[i] = new double[layer_sz[i+1]];

        // After Activation
        afterActivation[i] = new double[layer_sz[i+1]];

        // Delta
        delta[i] = new double[layer_sz[i+1]];
    }
}

// ================== Setup ==================

void init(){
    srand(time(0));
    init_parallel_for();
    init_network();
}

// ================== Data ==================

Data_Entry* getCircleData(int n, int w, int h, double x, double y, double r){
    Data_Entry* res = new Data_Entry[n];
    for(int i = 0; i < n; i++){
        double cx = (rand() % 1000) * (double)w / 1000;
        double cy = (rand() % 1000) * (double)h / 1000;

        double dist = sqrt((cx-x)*(cx-x) + (cy-y)*(cy-y));

        Vec input = new double[2];
        input[0] = cx;
        input[1] = cy;
        Vec output = new double[2];
        output[0] = dist <= r ? 1.0 : 0.0;
        output[1] = dist > r ? 1.0 : 0.0;

        res[i] = {input, output};
    }
    return res;
}

int main(){

    init();

    // training data
    // 1000 data points, circle with radius 3, center at (5,5)
    int training_n = 1000;
    Data_Entry* training_data = getCircleData(training_n, 10, 10, 5, 5, 3);
    // testing data
    int testing_n = 10;
    Data_Entry* testing_data = getCircleData(testing_n, 10, 10, 5, 5, 3);

    double lr = 1;
    int epochs = 100;

    while(true){
        auto startTime = chrono::high_resolution_clock::now();
        train(training_data, training_n, epochs, lr);
        auto endTime = chrono::high_resolution_clock::now();
        auto trainingTime = chrono::duration_cast<chrono::milliseconds>(endTime-startTime).count();

        cout << "took " << trainingTime << " milliseconds" << endl;
        cout << "lr: " << lr << endl;
        for(int i = 0; i < 10; i++){
            Data_Entry entry = testing_data[i];
            Vec input = entry.first;
            Vec output = entry.second;
            Vec res = forward(input);
            cout << "Input: " << to_string(input, layer_sz[0]-1);
            cout << " | Expected: " << to_string(output, layer_sz[layer_n]);
            cout << " | Got: " << to_string(res, layer_sz[layer_n]) << endl;
        }
        for(int i = 0; i < 100; i++) cout << "=";
        cout << endl;
    }

    return 0;
}