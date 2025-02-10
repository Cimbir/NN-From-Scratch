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

#define Vec vector<double>
#define Mat vector<Vec>

#define THREAD_AMOUNT 6

#define PARALLEL_SMALL_BACK 0
#define PARALLEL_BIG_BACK 0

// ================== Utils ==================

struct ParForData {
    int from;
    int to;
    function<void(int)> f;
};

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

// ================== Data Structure ==================

double operator*(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) {
        string error = "Vectors must be of the same length : " + std::to_string(a.size()) + " != " + std::to_string(b.size());
        throw invalid_argument(error);
    }
    double result = 0;
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

string to_string(Vec& vec) {
    string res = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        res += to_string(vec[i]);
        if (i < vec.size() - 1) {
            res += ", ";
        }
    }
    res += "]";
    return res;
}

// ================== Activation Functions ==================

class ActivationFunction {
public:
    virtual double operator()(double x) const = 0;
    virtual double derivative(double x, double y) const = 0;
    virtual unique_ptr<ActivationFunction> clone() const = 0;
    virtual ~ActivationFunction() = default;
};

class ReLU: public ActivationFunction {
public:
    double operator()(double x) const override {
        return max(0.0, x);
    }
    double derivative(double x, double y) const override {
        return x > 0 ? 1 : 0;
    }
    unique_ptr<ActivationFunction> clone() const override {
        return make_unique<ReLU>(*this);
    }
};

class Sigmoid: public ActivationFunction {
public:
    double operator()(double x) const override{
        return 1 / (1 + exp(-x));
    }
    double derivative(double x, double y) const override{
        return y * (1 - y);
    }
    unique_ptr<ActivationFunction> clone() const override {
        return make_unique<Sigmoid>(*this);
    }
};

// ================== Neural Network ==================

class Layer {
public:
    vector<Vec> weights;
    Layer() {}
    Layer(int input, int output) {
        this->weights = vector<Vec>(output, Vec(input));
        for(int i = 0; i < output; i++){
            for(int j = 0; j < input; j++){
                weights[i][j] = (rand() % 100) / 100.0;
            }
        }
    }

    Vec forward(Vec& input) {
        Vec res(weights.size());
        for(int i = 0; i < weights.size(); i++){
            res[i] = input * weights[i];
        }
        return res;
    }

    string asString() {
        string res = "[";
        for(int i = 0; i < weights.size(); i++){
            res += to_string(weights[i]);
            if(i < weights.size() - 1){
                res += ",\n";
            }
        }
        res += "]";
        return res;
    }
};

class ActivationLayer{
public:
    unique_ptr<ActivationFunction> activation;

    Vec before;
    Vec after;
    ActivationLayer() {}
    ActivationLayer(int input, unique_ptr<ActivationFunction> activation) {
        this->activation = move(activation);
        this->before = Vec(input);
        this->after = Vec(input);
    }

    Vec forward(Vec& input) {
        before = input;
        for(int i = 0; i < input.size(); i++){
            after[i] = (*activation)(input[i]);
        }
        return after;
    }
    Vec backward(Vec& actual, Vec& target) {
        Vec delta(actual.size());
        if(PARALLEL_SMALL_BACK){
            parallel_for(actual.size(), [&](int i){
                delta[i] = activation->derivative(before[i], actual[i]) * (target[i] - actual[i]);
            });
        }else{
            for(int i = 0; i < actual.size(); i++){
                delta[i] = activation->derivative(before[i], actual[i]) * (target[i] - actual[i]);
            }
        }
        return delta;
    }
    Vec backward(Vec& nextDelta, Layer& weights) {
        Vec delta(before.size());
        if(PARALLEL_SMALL_BACK){
            parallel_for(before.size(), [&](int i){
                delta[i] = 0;
                for(int j = 0; j < nextDelta.size(); j++){
                    delta[i] += nextDelta[j] * weights.weights[j][i];
                }
                delta[i] *= activation->derivative(before[i], after[i]);
            });
        }else{
            for(int i = 0; i < before.size(); i++){
                delta[i] = 0;
                for(int j = 0; j < nextDelta.size(); j++){
                    delta[i] += nextDelta[j] * weights.weights[j][i];
                }
                delta[i] *= activation->derivative(before[i], after[i]);
            }
        }
        return delta;
    }
};



struct NeuralNetwork {
    vector<Layer> layers;
    vector<ActivationLayer> activations;
    double lr;
    int passedEpochs = 0;

    NeuralNetwork(vector<int> sizes, unique_ptr<ActivationFunction> activation, double lr) {
        this->layers = vector<Layer>(sizes.size() - 1);
        for(int i = 0; i < layers.size(); i++){
            layers[i] = Layer(sizes[i], sizes[i + 1]);
        }

        this->activations = vector<ActivationLayer>(sizes.size() - 1);
        for(int i = 0; i < activations.size(); i++){
            activations[i] = ActivationLayer(sizes[i + 1], activation->clone());
        }
        this->lr = lr;
    }

    Vec forward(Vec input) {
        for(int i = 0; i < layers.size(); i++){
            input = layers[i].forward(input);
            input = activations[i].forward(input);
        }
        return input;
    }

    void backward(Vec& input, Vec& target) {
        Vec output = forward(input);

        int n = layers.size();
        vector<Vec> deltas(n);
        for(int i = 0; i < n; i++){
            deltas[i] = Vec(layers[i].weights.size());
        }

        deltas[n-1] = activations[n-1].backward(output, target);
        for(int i = layers.size() - 2; i >= 0; i--){
            deltas[i] = activations[i].backward(deltas[i+1], layers[i + 1]);
        }

        if(PARALLEL_BIG_BACK){
            parallel_for(layers.size(), [&](int i){
                for(int j = 0; j < layers[i].weights.size(); j++){
                    for(int k = 0; k < layers[i].weights[j].size(); k++){
                        if(i == 0){
                            layers[i].weights[j][k] += lr * deltas[i][j] * input[k];
                        }else{
                            layers[i].weights[j][k] += lr * deltas[i][j] * activations[i-1].after[k];
                        }
                    }
                }
            });
        }else{
            for(int i = 0; i < layers.size(); i++){
                for(int j = 0; j < layers[i].weights.size(); j++){
                    for(int k = 0; k < layers[i].weights[j].size(); k++){
                        if(i == 0){
                            layers[i].weights[j][k] += lr * deltas[i][j] * input[k];
                        }else{
                            layers[i].weights[j][k] += lr * deltas[i][j] * activations[i-1].after[k];
                        }
                    }
                }
            }
        }
    }

    void train(vector<Vec>& dataset, vector<Vec>& resultset, int epochs) {
        for(int i = 0; i < epochs; i++){
            cout << "epoch " << i << endl;
            for(int j = 0; j < dataset.size(); j++){
                backward(dataset[j], resultset[j]);
            }
            passedEpochs++;
            if(passedEpochs % 50 == 0){
                lr *= 0.99;
            }
        }
    }

};

pair<vector<Vec>,vector<Vec>> generateLineDataset() {
        vector<Vec> dataset = {
        {1, -1, 3},
        {1, -1, 1},
        {1, 1, -2},
        {1, 2, -3},
        {1, 2, 6},
        {1, 6, 2},
        {1, 6, -1}
    };

    vector<Vec> resultset = {
        {1, 0},
        {1, 0},
        {1, 0},
        {1, 0},
        {0, 1},
        {0, 1},
        {0, 1}
    };

    return {dataset, resultset};
}

pair<vector<Vec>,vector<Vec>> generateCircleDataset() {
    vector<Vec> dataset;
    vector<Vec> resultset;
    for(int i = 0; i < 1000; i++){
        double x = (rand() % 1000) / 100.0;
        double y = (rand() % 1000) / 100.0;
        dataset.push_back({1, x, y});
        resultset.push_back((x-5) * (x-5) + (y-5) * (y-5) < 9 ? Vec{1.0, 0.0} : Vec{0.0, 1.0});
    }
    return {dataset, resultset};
}

int main(){
    init();

    srand(time(0));
    auto [dataset, resultset] = generateCircleDataset();

    unique_ptr<ActivationFunction> sigmoid = make_unique<Sigmoid>();
    NeuralNetwork nn({3, 300, 300, 300, 2}, move(sigmoid), 0.1);

    while(true){
        cout << "training..." << endl;
        nn.train(dataset, resultset, 100);
        for(int i = 0; i < 10; i++){
            double x = dataset[i][1];
            double y = dataset[i][2];
            double dist = (x-5) * (x-5) + (y-5) * (y-5);
            Vec res = nn.forward(dataset[i]);
            cout << "( " << dataset[i][1] << ", " << dataset[i][2] << " )  " << dist << " -> " << to_string(res) << endl;
        }
        cout << "=====================" << endl;
    }

    return 0;
}