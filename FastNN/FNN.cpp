#include "FNN.hpp"
#include <cmath>

// ================== Utils ==================

string to_string(Vec v, int n){
    string res = "[";
    for(int i = 0; i < n; i++){
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

// ================== Neural Network Class ==================

FNN::FNN(int layer_n, int* layer_sz, int activation, double lr){
    this->layer_n = layer_n;
    this->layer_sz = layer_sz;
    this->act_type = activation;
    this->lr = lr;

    init();
}

void FNN::init(){
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



double FNN::activation(double x){
    if(act_type == _relu) return relu(x);
    if(act_type == _sigmoid) return sigmoid(x);
    return 0;
}

double FNN::activation_d(double x, double y){
    if(act_type == _relu) return relu_d(x, y);
    if(act_type == _sigmoid) return sigmoid_d(x, y);
    return 0;
}



Vec FNN::add_bias(Vec v){
    Vec res = new double[layer_sz[0]];
    for(int i = 0; i < layer_sz[0]-1; i++){
        res[i] = v[i];
    }
    res[layer_sz[0]-1] = 1;
    return res;
}

Vec FNN::forward(Vec input) {
    input = add_bias(input);
    for(int i = 0; i < layer_n; i++){
        for(int j = 0; j < layer_sz[i+1]; j++){
            double sum = 0;
            for(int k = 0; k < layer_sz[i]; k++){
                sum += weights[i][j][k] * input[k];
            }
            beforeActivation[i][j] = sum;
            afterActivation[i][j] = activation(sum);
        }
        input = afterActivation[i];
    }
    return input;
}

void FNN::backward(Vec input, Vec result, double lr) {
    Vec output = forward(input);
    input = add_bias(input);

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

void FNN::train(Data_Entry* dataset, int n, int epochs, double& lr){
    for(int e = 0; e < epochs; e++){
        for(int i = 0; i < n; i++){
            backward(dataset[i].first, dataset[i].second, lr);
        }
        if(e % 50 == 0) lr *= 0.99;
    }
}