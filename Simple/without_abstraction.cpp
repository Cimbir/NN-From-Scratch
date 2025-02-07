#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>
#include <thread>
#include <mutex>
#include <functional>

using namespace std;

// ================== Global Variables ==================

// Neural Network Architecture
#define Vec vector<double>
#define Mat vector<Vec>
#define Net vector<Mat>

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
vector<int> sizes = {2,20,20,20,2};
int layer_n = sizes.size()-1;

Net weights(layer_n);

vector<Vec> beforeActivation(layer_n);
vector<Vec> afterActivation(layer_n);

vector<Vec> delta(layer_n);

// Activation

#define _relu 0
#define _sigmoid 1
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
Vec forward(Vec input);
void backward(Vec input, Vec result, double lr);
void train(vector<Data_Entry>& dataset, int epochs, double& lr);
void init_network();

// Setup
void init();

// Data
vector<Data_Entry> getCircleData(int n, int w, int h, double x, double y, double r);

// ================== Utils ==================

string to_string(Vec& v){
    string res = "[";
    for(int i = 0; i < v.size(); i++){
        res += to_string(v[i]);
        if(i < v.size() - 1){
            res += ", ";
        }
    }
    res += "]";
    return res;
}

string to_string(Mat& m){
    string res = "[";
    for(int i = 0; i < m.size(); i++){
        res += to_string(m[i]);
        if(i < m.size() - 1){
            res += ",\n";
        }
    }
    res += "]";
    return res;
}

string to_string(Net& n){
    string res = "[";
    for(int i = 0; i < n.size(); i++){
        res += to_string(n[i]);
        if(i < n.size() - 1){
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

Vec forward(Vec input) {
    input.push_back(1);    
    for(int i = 0; i < weights.size(); i++){
        Mat& w = weights[i];
        Vec new_res(w.size());
        for(int j = 0; j < w.size(); j++){
            double sum = 0;
            for(int k = 0; k < w[j].size(); k++){
                sum += w[j][k] * input[k];
            }
            beforeActivation[i][j] = sum;
            new_res[j] = activation(sum);
            afterActivation[i][j] = new_res[j];
        }
        input = new_res;
    }
    return input;
}

void backward(Vec input, Vec result, double lr = 1) {
    Vec output = forward(input);
    input.push_back(1);

    // Update deltas
    for(int i = 0; i < delta[layer_n-1].size(); i++){
        delta[layer_n-1][i] = activation_d(beforeActivation[layer_n-1][i], afterActivation[layer_n-1][i]) * (result[i] - output[i]);
    }
    if(PARALLEL_DELTA_UPDATE){
        parallel_for(layer_n-1, [&](int i){
            for(int j = 0; j < delta[i].size(); j++){
                double sum = 0;
                for(int k = 0; k < delta[i+1].size(); k++){
                    sum += delta[i+1][k] * weights[i+1][k][j];
                }
                delta[i][j] = activation_d(beforeActivation[i][j], afterActivation[i][j]) * sum;
            }
        });
    }else{
        for(int i = layer_n-2; i >= 0; i--){
            for(int j = 0; j < delta[i].size(); j++){
                double sum = 0;
                for(int k = 0; k < delta[i+1].size(); k++){
                    sum += delta[i+1][k] * weights[i+1][k][j];
                }
                delta[i][j] = activation_d(beforeActivation[i][j], afterActivation[i][j]) * sum;
            }
        }
    }

    // Update weights
    if(PARALLEL_WEIGHT_UPDATE){
        parallel_for(layer_n, [&](int i){
            for(int j = 0; j < weights[i].size(); j++){
                for(int k = 0; k < weights[i][j].size(); k++){
                    if(i == 0){
                        weights[i][j][k] += lr * delta[i][j] * input[k];
                    }else{
                        weights[i][j][k] += lr * delta[i][j] * afterActivation[i-1][k];
                    }
                }
            }
        });
    }else{
        for(int i = 0; i < layer_n; i++){
            for(int j = 0; j < weights[i].size(); j++){
                for(int k = 0; k < weights[i][j].size(); k++){
                    if(i == 0){
                        weights[i][j][k] += lr * delta[i][j] * input[k];
                    }else{
                        weights[i][j][k] += lr * delta[i][j] * afterActivation[i-1][k];
                    }
                }
            }
        }
    }
}

void train(vector<Data_Entry>& dataset, int epochs, double& lr) {
    for(int i = 0; i < epochs; i++){
        for(int j = 0; j < dataset.size(); j++){
            backward(dataset[j].first, dataset[j].second, lr);
        }
        if(i % 50 == 0)
            lr *= 0.99;
    }
}

void init_network() {
    for(int i = 0; i < layer_n; i++){
        // Weights
        weights[i] = Mat(sizes[i+1]);
        for(int j = 0; j < sizes[i+1]; j++){
            // (i==0) is for bias
            int len = sizes[i]+(i==0);
            weights[i][j] = Vec(len);
            for(int k = 0; k < len; k++){
                weights[i][j][k] = (rand() % 100) / 100.0;
            }
        }

        // Before Activation
        beforeActivation[i] = Vec(sizes[i+1]);

        // After Activation
        afterActivation[i] = Vec(sizes[i+1]);

        // Delta
        delta[i] = Vec(sizes[i+1]);
    }
}

// ================== Setup ==================

void init(){
    srand(time(0));
    init_parallel_for();
    init_network();
}

// ================== Data ==================

vector<Data_Entry> getCircleData(int n, int w, int h, double x, double y, double r){
    vector<Data_Entry> res(n);
    for(int i = 0; i < n; i++){
        double cx = (rand() % 1000) * (double)w / 1000;
        double cy = (rand() % 1000) * (double)h / 1000;

        double dist = sqrt((cx-x)*(cx-x) + (cy-y)*(cy-y));

        Vec input = {cx, cy};
        Vec output = {dist <= r ? 1.0 : 0.0, dist > r ? 1.0 : 0.0};

        res[i] = {input, output};
    }
    return res;
}

int main(){

    init();

    // training data
    // 1000 data points, circle with radius 3, center at (5,5)
    vector<Data_Entry> training_data = getCircleData(1000, 10, 10, 5, 5, 3);
    // testing data
    vector<Data_Entry> test_data = getCircleData(100, 10, 10, 5, 5, 3);

    double lr = 0.1;

    while(true){
        auto startTime = chrono::high_resolution_clock::now();
        train(training_data, 100, lr);
        auto endTime = chrono::high_resolution_clock::now();
        auto trainingTime = chrono::duration_cast<chrono::milliseconds>(endTime-startTime).count();

        cout << "took " << trainingTime << " milliseconds" << endl;

        for(int i = 0; i < 10; i++){
            Data_Entry& entry = test_data[i];
            Vec& input = entry.first;
            Vec& output = entry.second;
            Vec res = forward(input);
            cout << "Input: " << to_string(input);
            cout << " | Expected: " << to_string(output);
            cout << " | Got: " << to_string(res) << endl;
        }
        for(int i = 0; i < 100; i++) cout << "=";
        cout << endl;
    }

    return 0;
}